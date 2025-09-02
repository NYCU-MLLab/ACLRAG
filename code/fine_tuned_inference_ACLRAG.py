import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
import sys
from safetensors.torch import load_file
from training_utils import tokenize_hotpot_combined
from datasets import load_dataset

# Set the computation device
device = "cuda"

# Parse model, data, and training arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Define Lora configuration
lora_config = LoraConfig(
    r=512,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model_args.train = False 
# Initialize model and send it to CUDA device
model = ICAE(model_args, training_args, lora_config)

# Load the fine-tuned checkpoint
# state_dict = torch.load(training_args.output_dir, map_location="cuda:0", weights_only=False)

training_args.output_dir = "trainer_2stage_Hotpot_CE+ACL_128m_0711/checkpoint-160000/pytorch_model.bin"
# training_args.output_dir = "../model/mistral_7b_ft_icae.safetensors"

print(f"Loading trained checkpoint from {training_args.output_dir}")
if training_args.output_dir.endswith(".pt") or training_args.output_dir.endswith(".bin"):
    state_dict = torch.load(training_args.output_dir, map_location="cuda:0")
else:
    state_dict = load_file(training_args.output_dir)
    
model.load_state_dict(state_dict, strict=False) # only load lora and memory token embeddings

model = model.to(device)
print("Model loaded successfully.")

file = "../../../hotpot/hotpot_dev_distractor_v1_ACL_tokenized_NoContextMax.jsonl"
lines = None
with open(file, "r") as f:
    lines = f.readlines()

# Prepare the model for 
model.eval()
max_out_length = 128


with torch.no_grad():
    with open("Hotpot_test/CLRAG_Hotpot_NoContextMax.jsonl", "w") as f:
        for line in tqdm(lines):
            data = json.loads(line)

            tokenized_context = data['context_combined_ids']
            tokenized_query = data['question_ids']
            # tokenized_context = model.tokenizer(full_irrelevant_context, truncation=True, max_length=5120, padding=False, return_attention_mask=False, add_special_tokens=False)
            # tokenized_query = model.tokenizer(data['question'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
            
            context_ids = torch.LongTensor([tokenized_context]).to(device)
            query_ids = torch.LongTensor([tokenized_query]).to(device)
            
            tokenized_input = torch.cat([torch.tensor([[model.bos_id]], device=device), query_ids, context_ids], dim=1).to(device)
            
            # tokenized_query = tokenized_query.to(device)

            memory_slots = model._compress(tokenized_input)

            decoder_input_ids = torch.cat([torch.tensor([[model.cl_token_id]], device=device), query_ids], dim=1)
            decoder_input_embeddings = model.tokens_to_embeddings(decoder_input_ids).to(device)

            memory_slots = memory_slots.to(decoder_input_embeddings)
            decoder_input_embeddings = torch.cat([memory_slots.unsqueeze(0), decoder_input_embeddings], dim=1)

            output = decoder_input_embeddings.clone()
            generate_text = []
            past_key_values = None
            for i in range(max_out_length):
                with model.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                # out = decoder(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :model.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                # print(next_token_id)
                
                if next_token_id.item() == model.eos_id:   # eos
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(generate_text)



            generated_text = model.tokenizer.decode(generate_text)

            # Structure output data
            output_ = {
                "output": generated_text
            }
            print(generated_text)
            f.write(json.dumps(output_) + "\n")

            torch.cuda.empty_cache()
