# ICAE that supports multi span concat

import transformers
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
import time
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from contextlib import contextmanager
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    # model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    # model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    # model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-3B-Instruct")
    lora_r: int = field(
        default=512,
        # default=128,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        # default=28000,
        default=5120,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self._keys_to_ignore_on_save = None
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, use_flash_attention_2=True, resume_download=True).to("cuda:0")
        
        self.training = self.model_args.train    
        
        if self.training:    # indepedent model for gradient checkpointing
            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, use_flash_attention_2=True, resume_download=True).to("cuda:0")
            
         
        
        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)
        self.alpha = 0.5

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.cl_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        
        # self.sep_token_id = self.vocab_size_with_mem + 3    # [SEP] token

        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=device).unsqueeze(0)    # mem tokens
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        # for contrstive learning
        self.state_dict_reconstruction = torch.load("trainer_pretrain_output_0430_128m_512r/checkpoint-500000/pytorch_model.bin", map_location="cuda:0")
        # self.state_dict_reconstruction = None
        self.temperature = 0.07  
        self.lora_reconstruction = None
        if self.training:
            self.init()


    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")

            # for pretrain
            # state_dict = load_file(self.training_args.restore_from)
            
            # for finetune
            state_dict = torch.load(self.training_args.restore_from, map_location="cuda:0")
            
            # # 訓練不同壓縮率 embedder size不同
            # keys_to_remove = [
            #     "memory_token_embed.weight",
            #     "icae.base_model.model.model.embed_tokens.weight",
            #     "icae.base_model.model.lm_head.weight",
            # ]

            # # 移除
            # for k in keys_to_remove:
            #     if k in state_dict:
            #         print(f"Removing {k} from checkpoint.")
            #         del state_dict[k]

            self.load_state_dict(state_dict, strict = False) # only load lora and memory token embeddings
            
            print(f"Finished loading from {self.training_args.restore_from}")


        print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))
        return num_segments


    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     prompt_answer_ids: torch.LongTensor = None,
    #     labels: Optional[torch.LongTensor] = None,):

    #     # encoder part
    #     batch_size = input_ids.size(0)
    #     total_length = input_ids.size(1)
    #     num_segments = self.compute_num_segments(total_length)
    #     # num_segments = 1
    #     segment_length = math.ceil(total_length / num_segments)
        
    #     prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
    #     max_compressed_length = num_segments * self.mem_size
    #     compress_outputs = torch.zeros((max_compressed_length, self.dim)).to(prompt_answer_embs)
        
    #     for segment_idx in range(num_segments):
            
    #         start_idx = segment_idx * segment_length
    #         end_idx = min((segment_idx + 1) * segment_length, total_length)
    #         segment_input_ids = input_ids[:, start_idx:end_idx]
    #         segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
    #         mem_flag = segment_input_ids >= self.vocab_size

    #         segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
    #         segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

    #         # compress the current segment
    #         segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
    #         segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

    #         # collect memory tokens
    #         compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
    #         del segment_input_ids, segment_input_embedding
    #         torch.cuda.empty_cache()
            
    #     # decoder part
    #     decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens

    #     prompt_answer_embs[decoder_mem_flag] = compress_outputs  # replace memory slots
    #     special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
    #     prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
        
    #     if self.training:   # has an independent se.f.decoder
    #         decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
    #     else:
    #         with self.icae.disable_adapter():   # no independent decoder; use self.icae
    #             decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)


    #     logits = decoder_outputs.logits
    #     effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
    #     target_ids = labels[:,1:].reshape(-1)
    #     loss = self.loss_fct(effective_logits, target_ids)
        
    #     return {"loss": loss, "logits": logits}
    
    # CE_loss_log = []
    # contrastive_loss_log = []
    
    # def forward(self, **inputs):
    #     # print(inputs.keys())
    #     # torch.autograd.set_detect_anomaly(True)
    #     # output 
    #     input_ids = inputs["input_ids"]
    #     prompt_answer_ids = inputs["prompt_answer_ids"]
    #     labels_ids = inputs["label_ids"]
    #     relevant_context_ids = inputs["relevant_context_ids"]
    #     irrelevant_context_ids_list = inputs["irrelevant_context_ids"]
    #     answer_list = inputs["answer_ids"]
        
        
    #     input_ids = torch.cat([input_ids, self.append_sequence], dim=1)
    #     # relevant_context_ids = torch.tensor(relevant_context_ids).to(device=self.append_sequence.device)
    #     # print("relevant_context_ids", relevant_context_ids.shape)
    #     # print(relevant_context_ids)
    #     relevant_context_ids = torch.cat([relevant_context_ids, self.append_sequence], dim=1)
    #     irrelevant_context_ids_list = [
    #         torch.cat([
    #             torch.tensor(ids if isinstance(ids[0], int) else ids[0], device=self.append_sequence.device),
    #             self.append_sequence.squeeze(0)
    #         ], dim=0)
    #         for ids in irrelevant_context_ids_list
    #     ]

    #     mem_flag_anchor = input_ids >= self.vocab_size
    #     mem_flag_relevant = relevant_context_ids >= self.vocab_size
    #     mem_flag_irrelevant = [ids >= self.vocab_size for ids in irrelevant_context_ids_list]

        
    #     with self.with_partial_state_dict(self.icae, self.state_dict_reconstruction):
    #         with torch.no_grad():
    #         # 先生成正負樣本
    #             relevant_context_embedding = self.tokens_to_embeddings(relevant_context_ids)
    #             irrelevant_context_embedding = []
    #             for ids, mem_flag in zip(irrelevant_context_ids_list, mem_flag_irrelevant):
    #                 emb = self.tokens_to_embeddings(ids)
    #                 irrelevant_context_embedding.append(emb)
        
    #             positive_outputs = self.icae(inputs_embeds=relevant_context_embedding, output_hidden_states=True)
    #             pos_rep = positive_outputs.hidden_states[-1]
    #             relevant_slots = pos_rep[mem_flag_relevant]

    #             irrelevant_slots = []
    #             for emb, mem_flag in zip(irrelevant_context_embedding, mem_flag_irrelevant):
    #                 output = self.icae(inputs_embeds=emb.unsqueeze(0), output_hidden_states=True)
    #                 rep = output.hidden_states[-1].squeeze(0)
    #                 irrelevant_slots.append(rep[mem_flag])

    #     # anchor（要反向傳播）
    #     input_embedding = self.tokens_to_embeddings(input_ids)
    #     anchor_outputs = self.icae(inputs_embeds=input_embedding, output_hidden_states=True)
    #     anchor_rep = anchor_outputs.hidden_states[-1]
    #     anchor_slots = anchor_rep[mem_flag_anchor]

    #     # pooling
    #     anchor_vec = anchor_slots.mean(dim=0, keepdim=True)
    #     positive_vec = relevant_slots.mean(dim=0, keepdim=True)
    #     negative_vecs = torch.cat(
    #         [slots.mean(dim=0, keepdim=True) for slots in irrelevant_slots],
    #         dim=0
    #     )

    #     # normalize
    #     anchor_vec = torch.nn.functional.normalize(anchor_vec, dim=-1)
    #     positive_vec = torch.nn.functional.normalize(positive_vec, dim=-1)
    #     negative_vecs = torch.nn.functional.normalize(negative_vecs, dim=-1)

    #     # similarity
    #     pos_sim = torch.matmul(anchor_vec, positive_vec.T)
    #     neg_sim = torch.matmul(anchor_vec, negative_vecs.T)
    #     logits = torch.cat([pos_sim, neg_sim], dim=1)
    #     labels = torch.zeros(1, dtype=torch.long).to(logits.device)

    #     contrastive_loss = torch.nn.functional.cross_entropy(logits / self.temperature, labels)
    #     all_CE_losses = []
    #     # print("answer_list", len(answer_list))
    #     # print("answer_list", answer_list)
    #     for answer in answer_list[0]:
    #         # print("answer", len(answer))
    #         # print("answer", answer)
    #         # decoder part
    #         # decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens
    #         eos_tensor = torch.tensor([self.eos_id], dtype=torch.long).to(logits.device)
    #         eos_tensor = eos_tensor.unsqueeze(1) 
    #         # prompt_answer_ids = torch.tensor(prompt_answer_ids).to(logits.device)
    #         answer = torch.tensor(answer).to(logits.device)
    #         answer = answer.unsqueeze(0) 
    #         prompt_answer = torch.cat([prompt_answer_ids, answer, eos_tensor], dim=1)
    #         # print("prompt_answer", prompt_answer.shape)
    #         # print(prompt_answer)
    #         decoder_mem_flag = (prompt_answer >= self.vocab_size) & (prompt_answer < self.vocab_size + self.mem_size)   # only mem tokens
    #         prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer)

    #         prompt_answer_embs[decoder_mem_flag] = anchor_slots  # replace memory slots
    #         special_prompt = prompt_answer >= self.vocab_size_with_mem
    #         prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer[special_prompt] - self.vocab_size).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
            
    #         if self.training:   # has an independent se.f.decoder
    #             decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
    #         else:
    #             with self.icae.disable_adapter():   # no independent decoder; use self.icae
    #                 decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)


    #         logits = decoder_outputs.logits
    #         effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
    #         # input["answers"] = [[794,125,0...] ,[....]]

        
    #         # prompt_answer_ids appened answer + [128009](EOS)
    #         # prompt_answer = prompt_answer_ids + answer + [self.eos_id]  # append the answer and EOS token
    #         # labels appened answer + [128009](EOS)
    #         # labels_ids = torch.tensor(labels_ids).to(logits.device)
    #         labels = torch.cat([labels_ids, answer, eos_tensor], dim=1) # append the answer and EOS token
    #         target_ids = labels[:,1:].reshape(-1) #do teacher forcing
    #         crossEntropy_loss = self.loss_fct(effective_logits, target_ids)
    #         all_CE_losses.append(crossEntropy_loss)
        
    #     CE_loss = torch.min(torch.stack(all_CE_losses))
    #     # 清理 GPU 記憶體

    #     # print("CE Loss: ", CE_loss.item())
    #     self.CE_loss_log.append(CE_loss.item())
    #     # print("Contrastive Loss: ", contrastive_loss.item())
    #     self.contrastive_loss_log.append(contrastive_loss.item())
    #     torch.cuda.empty_cache()
        
        
    #     return {"loss": CE_loss + contrastive_loss}


    def forward(self, **inputs):
        question_ids = inputs["question_ids"]
        answer_ids = inputs["answer_ids"]
        supporting_facts_ids = inputs["supporting_facts_ids"]
        context_combined_ids = inputs["context_combined_ids"]
        context_combined_mask = inputs["context_combined_mask"]

        bos_tensor = torch.tensor([self.bos_id], dtype=torch.long, device=question_ids.device)
        eos_tensor = torch.tensor([self.eos_id], dtype=torch.long, device=question_ids.device)
        vocab_size_tensor = torch.full((self.mem_size,), self.vocab_size, dtype=torch.long, device=question_ids.device)
        cl_tensor = torch.tensor([self.cl_token_id], dtype=torch.long, device=question_ids.device)

        input_ids = torch.cat([bos_tensor, question_ids, context_combined_ids], dim=0)
        prompt_answer_ids = torch.cat([vocab_size_tensor, cl_tensor, question_ids], dim=0)
        labels_ids = torch.full((prompt_answer_ids.size(0),), -100, dtype=torch.long, device=question_ids.device)

        if len(supporting_facts_ids) > 0:
            relevant_context_ids = torch.cat([bos_tensor] + supporting_facts_ids, dim=0)
        else:
            relevant_context_ids = bos_tensor

        irrelevant_context_ids_list = [[] for _ in range(10)]
        for token, mask in zip(context_combined_ids, context_combined_mask):
            if 0 <= mask < 10:
                irrelevant_context_ids_list[mask].append(token.item())
        irrelevant_context_ids_list = [[self.bos_id] + ids for ids in irrelevant_context_ids_list]

        input_ids = torch.cat([input_ids, self.append_sequence.squeeze(0)], dim=0)
        relevant_context_ids = torch.cat([relevant_context_ids, self.append_sequence.squeeze(0)], dim=0)
        irrelevant_context_ids_list = [
            torch.cat([
                torch.tensor(ids if isinstance(ids[0], int) else ids[0], device=self.append_sequence.device),
                self.append_sequence.squeeze(0)
            ], dim=0)
            for ids in irrelevant_context_ids_list
        ]

        mem_flag_anchor = input_ids >= self.vocab_size
        mem_flag_relevant = relevant_context_ids >= self.vocab_size
        mem_flag_irrelevant = [ids >= self.vocab_size for ids in irrelevant_context_ids_list]

        with self.with_partial_state_dict(self.icae, self.state_dict_reconstruction):
            with torch.no_grad():
                relevant_context_embedding = self.tokens_to_embeddings(relevant_context_ids).unsqueeze(0)
                irrelevant_context_embedding = [self.tokens_to_embeddings(ids) for ids in irrelevant_context_ids_list]

                pos_rep = self.icae(inputs_embeds=relevant_context_embedding, output_hidden_states=True).hidden_states[-1].squeeze(0)
                relevant_slots = pos_rep[mem_flag_relevant]

                irrelevant_slots = []
                for emb, mem_flag in zip(irrelevant_context_embedding, mem_flag_irrelevant):
                    rep = self.icae(inputs_embeds=emb.unsqueeze(0), output_hidden_states=True).hidden_states[-1].squeeze(0)
                    irrelevant_slots.append(rep[mem_flag])

        input_embedding = self.tokens_to_embeddings(input_ids).unsqueeze(0)
        anchor_rep = self.icae(inputs_embeds=input_embedding, output_hidden_states=True).hidden_states[-1].squeeze(0)
        anchor_slots = anchor_rep[mem_flag_anchor]
        anchor_vec = torch.nn.functional.normalize(anchor_slots.mean(dim=0, keepdim=True), dim=-1)

        positive_vec = torch.nn.functional.normalize(relevant_slots.mean(dim=0, keepdim=True), dim=-1)
        negative_vecs = torch.nn.functional.normalize(torch.cat([s.mean(dim=0, keepdim=True) for s in irrelevant_slots], dim=0), dim=-1)

        pos_sim = torch.matmul(anchor_vec, positive_vec.T)
        neg_sim = torch.matmul(anchor_vec, negative_vecs.T)
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(1, dtype=torch.long).to(logits.device)
        contrastive_loss = torch.nn.functional.cross_entropy(logits / self.temperature, labels)

        # === FGSM for anchor ===
        anchor_embeds = input_embedding.detach().clone().requires_grad_(True)
        dummy_label = torch.zeros(1, dtype=torch.long).to(anchor_embeds.device)
        logits_anchor = self.icae(inputs_embeds=anchor_embeds).logits[:, -1, :]
        adv_loss_anchor = self.loss_fct(logits_anchor, dummy_label)
        adv_loss_anchor.backward()

        epsilon = 1e-3
        perturb_anchor = epsilon * anchor_embeds.grad / (anchor_embeds.grad.norm(p=2) + 1e-8)
        perturbed_anchor = anchor_embeds + perturb_anchor
        perturbed_rep = self.icae(inputs_embeds=perturbed_anchor, output_hidden_states=True).hidden_states[-1].squeeze(0)
        perturbed_anchor_slots = perturbed_rep[mem_flag_anchor]
        anchor_vec_adv = torch.nn.functional.normalize(perturbed_anchor_slots.mean(dim=0, keepdim=True), dim=-1)

        # === FGSM for context ===
        context_tensor = context_combined_ids.unsqueeze(0)
        context_embeds = self.tokens_to_embeddings(context_tensor).detach().clone().requires_grad_(True)
        logits_context = self.icae(inputs_embeds=context_embeds).logits[:, -1, :]
        adv_loss_context = self.loss_fct(logits_context, dummy_label)
        adv_loss_context.backward()

        perturb_context = epsilon * context_embeds.grad / (context_embeds.grad.norm(p=2) + 1e-8)
        perturbed_context = (context_embeds + perturb_context).squeeze(0)

        num_supports = len(supporting_facts_ids)
        relevant_splits = [[] for _ in range(num_supports)]
        irrelevant_splits = [[] for _ in range(10)]
        for i, mask_val in enumerate(context_combined_mask):
            token_embed = perturbed_context[i]
            if mask_val < 0:
                idx = -mask_val - 1
                if idx < num_supports:
                    relevant_splits[idx].append(token_embed)
            elif 0 <= mask_val < 10:
                irrelevant_splits[mask_val].append(token_embed)

        perturbed_relevant_context = [torch.stack(s, dim=0) for s in relevant_splits if s]
        perturbed_irrelevant_context_list = [torch.stack(s, dim=0) for s in irrelevant_splits if s]
        relevant_context_emb_adv = torch.cat(perturbed_relevant_context, dim=0) if perturbed_relevant_context else torch.empty(0, device=perturbed_context.device)

        positive_vec_adv = torch.nn.functional.normalize(relevant_context_emb_adv.mean(dim=0, keepdim=True), dim=-1)
        negative_vecs_adv = torch.nn.functional.normalize(torch.cat([s.mean(dim=0, keepdim=True) for s in perturbed_irrelevant_context_list], dim=0), dim=-1)

        pos_sim_adv = torch.matmul(anchor_vec_adv, positive_vec_adv.T)
        neg_sim_adv = torch.matmul(anchor_vec_adv, negative_vecs_adv.T)
        logits_adv = torch.cat([pos_sim_adv, neg_sim_adv], dim=1)
        adv_contrastive_loss = torch.nn.functional.cross_entropy(logits_adv / self.temperature, labels)

        # === Answer generation loss ===
        prompt_answer = torch.cat([prompt_answer_ids, answer_ids, eos_tensor], dim=0)
        decoder_mem_flag = (prompt_answer >= self.vocab_size) & (prompt_answer < self.vocab_size + self.mem_size)
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer)
        prompt_answer_embs[decoder_mem_flag] = anchor_slots
        special_prompt = prompt_answer >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer[special_prompt] - self.vocab_size).to(prompt_answer_embs)
        prompt_answer_embs = prompt_answer_embs.unsqueeze(0)

        if self.training:
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        labels = torch.cat([labels_ids, answer_ids, eos_tensor], dim=0)
        target_ids = labels[1:].reshape(-1)
        CE_loss = self.loss_fct(effective_logits, target_ids)

        torch.cuda.empty_cache()
        # total_loss = CE_loss + contrastive_loss + adv_contrastive_loss
        total_loss = CE_loss + adv_contrastive_loss

        return {"loss": total_loss}

    
    def save_loss_logs(self):
        # 儲存損失日誌到文本檔案
        with open("CE_loss_log.txt", "w") as f:
            f.write("CE Loss Log\n")
            for loss in self.CE_loss_log:
                f.write(f"{loss}\n")  # 將每個損失值寫入新的一行

        with open("contrastive_loss_log.txt", "w") as f:
            f.write("Contrastive Loss Log\n")
            for loss in self.contrastive_loss_log:
                f.write(f"{loss}\n")  # 將每個損失值寫入新的一行
    
    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)    # replace special token's embedding from self.memory_token_embed
        return embeddings
        
    @contextmanager
    def with_partial_state_dict(self, model, state_dict_partial):
        
        if state_dict_partial is None:
            raise ValueError("state_dict_partial cannot be None. Please check the initialization process.")

        # 備份原始 state_dict（deepcopy 是必要的，否則還原時會失敗）
        original_state = copy.deepcopy(model.state_dict())

        # 套用部分 state_dict（允許 mismatch）
        model.load_state_dict(state_dict_partial, strict=False)

        try:
            yield
        finally:
            # 還原原本參數
            model.load_state_dict(original_state, strict=True)

    
    def _compress(
        self,
        input_ids: torch.LongTensor = None):  # for inference; compress a fixed length of input into memory slots

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        # num_segments = self.compute_num_segments(total_length)
        # segment_length = math.ceil(total_length / num_segments)
        num_segments = 1
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        return compress_outputs
    