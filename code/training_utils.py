from transformers import Trainer
import os
import torch
import random

from transformers.trainer_utils import get_last_checkpoint
import math
from transformers import DataCollatorWithPadding # @LJK
from torch.nn.utils.rnn import pad_sequence
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):

    if (training_args.output_dir is None):
        print("Error: output_dir is None")
    else:
        print(f"Output directory: {training_args.output_dir}")
    

    if(training_args.resume_from_checkpoint is None):
        print("No checkpoint to resume from")
    else:
        print(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # if max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
    #     print("Warning: batch size is 1, which may lead to slow training speed. Consider increasing the batch size.")
    #     data_collator = None
        
    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    # data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer, pad_to_multiple_of=8)  # pad 到 8 的倍數可加速

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")
    # print(train_dataset[0].keys())  # 確保包含 'answer_ids'
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    # print("do eval")
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    # model.save_loss_logs()
    

def text_extraction(input_ids, length, lm_ratio=0.0):
    
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"
    
    # ae
    if random.random() >= lm_ratio: 
        if input_len <= length: # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start: random_start+length], []
    
    # lm    
    if input_len <= length:
        r = random.randint(0, input_len-1)
        return input_ids[:r+1], input_ids[r+1:]
    else: 
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length], input_ids[random_start+length:]


def pretrain_tokenize_function(examples, model, mem, lm_ratio=0.0):
    # text_output = model.tokenizer(examples["text"], truncation=False, padding=False, return_attention_mask=False)
    text_output = model.tokenizer(examples["text"], truncation=True, padding=False, return_attention_mask=False)
    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []

    max_len = model.training_args.model_max_length  # heuristic

    for idx in range(len(text_output["input_ids"])):
        
        ae = True
        a, b = text_extraction(text_output["input_ids"][idx], max_len, lm_ratio=lm_ratio)
        length_a = len(a)
        num_segments = model.compute_num_segments(length_a)
        total_mem_length = num_segments * model.mem_size
        
        if len(b) > model.training_args.min_tokens_for_lm:  # avoid too few tokens for lm, which is a waste of computing
            ae = False
            b = b[:max_len]

        text_output['input_ids'][idx] = a

        # decoder part: note that in v2, we add mem_tokens to the prompt_ids for easy implementation; which is different from v1 implementation where mem tokens are not in the prompt_ids
        if ae:  # autoencoding objective
            prompt_ids = [mem[0]] * total_mem_length + [model.ae_token_id]
            answer_ids = a + [model.eos_id]    # if ae, eos token
        else:   # lm objective
            prompt_ids = [mem[0]] * total_mem_length
            if model.training_args.add_special_token_for_lm:
                prompt_ids += [model.lm_token_id]
            answer_ids = b   # if lm, no eos token

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
        if ae:
            labels = [-100] * len(prompt_ids) + answer_ids
        else:
            labels = [-100] * len(prompt_ids) + [-100] * model.training_args.leave_tokens_for_lm + answer_ids[model.training_args.leave_tokens_for_lm:] # no loss for leave_tokens_for_lm
        text_output['labels'].append(labels)
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output


def instruct_ft_tokenize_function(examples, model, mem):

    text_output = model.tokenizer(examples["context"], max_length=5120, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)
    prompt_output = model.tokenizer(examples["question"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    label_output = model.tokenizer(examples["answer"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []

    max_len = model.training_args.model_max_length  # heuristic

    for idx in range(len(text_output["input_ids"])):
        
        length = len(text_output["input_ids"][idx])
        # num_segments = model.compute_num_segments(length)
        num_segments = 1
        total_mem_length = num_segments * model.mem_size
        
        prompt_ids = [mem[0]] * total_mem_length + [model.ft_token_id] + prompt_output['input_ids'][idx]
        # prompt_ids = [1, 733, 16289, 28793] + prompt_ids + [733, 28748, 16289, 28793]   # special formats for prompt in Mistral
        answer_ids = label_output['input_ids'][idx] + [model.eos_id]

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
            
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output['labels'].append(labels)
        
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output

def shuffle_retrieval_position(relevant_context_id, irrelevant_context_id, max_length):
    # 合併並打亂順序
    retrievals = [relevant_context_id] + irrelevant_context_id
    random.shuffle(retrievals)

    # 初始化結果
    output = []
    current_length = 0
    relevant_added = False
    relevant_start_idx = -1

    # 處理每個 ctx，去掉首個 128000
    retrievals = [ctx[1:] if ctx and ctx[0] == 128000 else ctx for ctx in retrievals]

    for ctx in retrievals:
        remaining_length = max_length - current_length

        # 如果是 relevant_context 且還未加入
        if ctx == relevant_context_id[1:] and not relevant_added:
            # 如果無法直接加入，開始刪除前面的 retrievals
            while len(ctx) + current_length > max_length and output:
                current_length -= len(output[0])
                output = output[1:]

            # 插入 relevant_context_id
            relevant_start_idx = len(output)  # 紀錄插入起始位置
            output.append(ctx)
            current_length += len(ctx)
            relevant_added = True

        # 其他 retrievals 正常處理
        elif current_length + len(ctx) <= max_length:
            output.append(ctx)
            current_length += len(ctx)

    # 如果 relevant_context_id 尚未加入，則在適當位置插入
    if not relevant_added:
        while len(relevant_context_id[1:]) + current_length > max_length and output:
            current_length -= len(output[0])
            output = output[1:]

        relevant_start_idx = len(output)  # 紀錄插入起始位置
        output.append(relevant_context_id[1:])

    # 將 output 中的 list 合併成單一 list 並加上 BOS
    flat_output = [128000]  # 在前面加上一次 BOS
    for ctx in output:
        flat_output.extend(ctx)

    # 紀錄 relevant_context_id 的長度
    relevant_length = len(relevant_context_id) - 1  # 因為已去除 BOS

    # 調整 start_idx 為 flat_output 中的實際索引
    if relevant_start_idx != -1:
        relevant_start_idx = sum(len(output[i]) for i in range(relevant_start_idx)) + 1  # +1 是因為加了 BOS

    return flat_output, relevant_start_idx, relevant_length

def contrastive_adversarial_tokenize_function(examples, model, mem):
    output = {}

    question_id = model.tokenizer(examples["question"], truncation=True, padding=False, return_attention_mask=False)
    output["question_ids"] = question_id["input_ids"]

    relevant_context_id = model.tokenizer(examples["ctxs"][0]["text"], max_length=5120, truncation=True, padding=False, return_attention_mask=False)
    output["relevant_context_ids"] = relevant_context_id["input_ids"]

    irrelevant_context_ids = []
    for i in range(1, len(examples["ctxs"])):
        irrelevant_context_ids.append((model.tokenizer(examples["ctxs"][i]["text"], truncation=True, padding=False, return_attention_mask=False))["input_ids"])
    output["irrelevant_context_ids"] = irrelevant_context_ids
    
    answer_ids = []
    for answer in examples["answers"]:
        answer_ids.append(model.tokenizer(answer, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)["input_ids"])
    output["answer_ids"] = answer_ids
    # print("answer_ids is added")

    shuffle_context_ids, start, length = shuffle_retrieval_position(output["relevant_context_ids"], output["irrelevant_context_ids"], 5000)
    
    prompt_ids = [mem[0]] * TrainingArguments.fixed_mem_size + [128386] + output["question_ids"][1:] # [128386]: [CL]
    output["input_ids"] = output["question_ids"] + shuffle_context_ids[1:]
    output["prompt_answer_ids"] = prompt_ids 
    output["label_ids"] = [-100] * len(prompt_ids) 

    
    return output

def tokenize_hotpot_example(example, model, mem):
    output = {}

    # 特殊 token ID 定義
    BOS_TOKEN_ID = 128000
    CL_TOKEN_ID = 128386
    SKIP_TOKEN_ID = -100

    # 1. tokenize question，加上 BOS
    question_ids = model.tokenizer(
        example["question"],
        truncation=True,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False
    )["input_ids"]
    question_ids = [BOS_TOKEN_ID] + question_ids

    # 2. tokenize relevant_context，加上 BOS
    relevant_ids = model.tokenizer(
        example["relevant_context"],
        truncation=True,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False
    )["input_ids"]
    relevant_ids = [BOS_TOKEN_ID] + relevant_ids

    # 3. tokenize irrelevant_context（共 9 段），每段加 BOS
    irrelevant_ids = []
    for ctx in example["irrelevant_context"]:
        ids = model.tokenizer(
            ctx,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False
        )["input_ids"]
        ids = [BOS_TOKEN_ID] + ids
        irrelevant_ids.append(ids)

    # 4. tokenize full context（不加 BOS）
    context_ids = model.tokenizer(
        example["context"],
        truncation=True,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False
    )["input_ids"]

    # 5. tokenize answer（hotpot 通常只有一個答案）
    answer_ids = [
        model.tokenizer(
            example["answer"],
            truncation=True,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False
        )["input_ids"]
    ]

    # 6. construct output fields
    output["query_ids"] = question_ids
    output["relevant_context_ids"] = relevant_ids
    output["irrelevant_context_ids"] = irrelevant_ids
    if len(irrelevant_ids) < 0:
        output["irrelevant_context_ids"].extend([[]] * (9 - len(irrelevant_ids)))
    output["context_ids"] = context_ids
    output["answer_ids"] = answer_ids

    # 7. input_ids = question + context
    output["input_ids"] = question_ids + context_ids

    # 8. prompt_answer_ids = mem[0]*128 + [CL] + question（去掉 query 的 BOS）
    prompt_answer_ids = [mem[0]] * TrainingArguments.fixed_mem_size + [CL_TOKEN_ID] + question_ids[1:]
    output["prompt_answer_ids"] = prompt_answer_ids

    # 9. labels = -100 * len(prompt)
    output["label_ids"] = [SKIP_TOKEN_ID] * len(prompt_answer_ids)

    return output

import random
from typing import Dict, Any
from transformers import PreTrainedTokenizer
def tokenize_hotpot_combined(example: Dict[str, Any], model, mem=None) -> Dict[str, Any]:
    MAX_TOTAL_LEN = 1000000000000
    RESERVED_FOR_PROMPT = 128  # 留給 question 使用的空間

    question = example["question"]
    passages = example["context"]
    supporting_sentences = example["supporting_facts"]
    fact_position = example["fact_position"]
    answer = example["answer"]

    answer_ids = model.tokenizer.encode(answer, add_special_tokens=False)
    question_ids = model.tokenizer.encode(question, add_special_tokens=False)
    sep_token = model.tokenizer.encode(" ", add_special_tokens=False)
    max_len = MAX_TOTAL_LEN - len(question_ids) - RESERVED_FOR_PROMPT
    # 預處理 passages 為三元組 (original_idx, passage, list_of_sentences)
    passage_info = []
    for idx, passage in enumerate(passages):
        sents = [s.strip() for s in passage.split('. ') if s.strip()]
        passage_info.append((idx, passage, sents))

    # 確保包含 supporting fact 的 passages 會保留
    support_indices = set(fact_position)
    support_passages = []
    support_length = 0
    for idx, passage, _ in passage_info:
        if idx in support_indices:
            ids = model.tokenizer.encode(passage, add_special_tokens=False)
            extra = len(ids) + (len(sep_token) if support_passages else 0)
            support_passages.append((idx, passage, ids))
            support_length += extra

    # 從其他 passage 中選可容納的
    optional_passages = []
    for idx, passage, _ in passage_info:
        if idx not in support_indices:
            ids = model.tokenizer.encode(passage, add_special_tokens=False)
            optional_passages.append((idx, passage, ids))
    random.shuffle(optional_passages)

    optional_selected = []
    optional_total_len = 0
    for idx, passage, ids in optional_passages:
        extra = len(ids) + (len(sep_token) if (support_passages or optional_selected) else 0)
        if support_length + optional_total_len + extra <= max_len:
            optional_selected.append((idx, passage, ids))
            optional_total_len += extra

    # 合併並打亂
    all_passages = support_passages + optional_selected
    random.shuffle(all_passages)

    context_ids = []
    context_mask = []
    for orig_idx, _, ids in all_passages:
        if context_ids:
            context_ids.extend(sep_token)
            context_mask.extend([orig_idx] * len(sep_token))
        context_ids.extend(ids)
        context_mask.extend([orig_idx] * len(ids))

    # 將 supporting_facts 編成 token ids，並加上空格（句與句之間）
    def interleave_with_space(sentences):
        ids = []
        for i, s in enumerate(sentences):
            ids.extend(model.tokenizer.encode(s, add_special_tokens=False))
            if i < len(sentences) - 1:
                ids.extend(sep_token)
        return ids

    supporting_ids_list = []
    for sent in supporting_sentences:
        ids = model.tokenizer.encode(sent, add_special_tokens=False)
        supporting_ids_list.append(ids)

    # 在 context 中尋找並標記 mask = -n
    def find_sublist(full, part):
        for i in range(len(full) - len(part) + 1):
            if full[i:i + len(part)] == part:
                return i
        return -1

    for idx, supp_ids in enumerate(supporting_ids_list):
        start = 0
        while start < len(context_ids):
            found = find_sublist(context_ids[start:], supp_ids)
            if found == -1:
                break
            abs_start = start + found
            abs_end = abs_start + len(supp_ids)
            context_mask[abs_start:abs_end] = [- (idx + 1)] * len(supp_ids)
            start = abs_end

    return {
        "question_ids": question_ids,
        "answer_ids": answer_ids,
        "supporting_facts_ids": supporting_ids_list,
        "context_combined_ids": context_ids,
        "context_combined_mask": context_mask,
    }

def custom_data_collator(features):
    batch = {}

    # 一般欄位（都是 list[int]），可以直接 pad 成 tensor
    keys_to_pad = ["input_ids", "prompt_answer_ids", "label_ids", "relevant_context_ids"]
    for key in keys_to_pad:
        if key in features[0]:
            # if not  key == "input_ids" :
            #     batch[key] = [torch.tensor(f[key]) for f in features] 
            # else: 
                batch[key] = pad_sequence(
                    [torch.tensor(f[key]) for f in features],
                    batch_first=True,
                    padding_value=128256  # 或 self.pad_token_id，看你模型需求
                )

    # 特殊欄位：irrelevant_context_ids 是 list of list of int，不 pad，保留原樣
    batch["irrelevant_context_ids"] = [f["irrelevant_context_ids"] for f in features]
    batch["answer_ids"] = [f["answer_ids"] for f in features]
    

    return batch


def simple_data_collator(features):
    f = features[0]  # batch size = 1

    return {
        "question_ids": torch.tensor(f["question_ids"]),
        "answer_ids": torch.tensor(f["answer_ids"]),
        "supporting_facts_ids": [torch.tensor(x) for x in f["supporting_facts_ids"]],
        "context_combined_ids": torch.tensor(f["context_combined_ids"]),
        "context_combined_mask": torch.tensor(f["context_combined_mask"]),
    }

    

class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        prompt_answer_ids = [torch.tensor(example["prompt_answer_ids"], dtype=torch.long) for example in examples]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self.dynamic_padding(labels)
        batch = {"input_ids": input_ids, "labels": labels, "prompt_answer_ids": prompt_answer_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences