import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import contrastive_adversarial_tokenize_function, custom_data_collator, train_model, tokenize_hotpot_example
import logging

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.train = True  
    print(model_args)
    print(data_args)
    
    training_args.bp16 = True
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.save_safetensors = False
    training_args.save_steps = 10000
    training_args.save_total_limit = 10
    training_args.restore_from = "trainer_pretrain_output_0430_128m_128r/checkpoint-500000/pytorch_model.bin"
    training_args.remove_unused_columns=False
    training_args.logging_steps = 50
    training_args.num_train_epochs = 2
    # training_args.max_steps = 90447*2
    # logging.basicConfig(level=logging.INFO)
    # training_args.evaluation_strategy = "steps"
    # training_args.eval_steps = 10

    
    training_args.output_dir = "trainer_2stage_Hotpot_CE+CL_32r_0701"

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    
    memory_size = training_args.fixed_mem_size

    train_file = "../../../hotpot/hotpot_train_v1.1_formatted.jsonl"

    print("Loading dataset...")

    dataset = load_dataset("json", data_files={"train": train_file}, streaming=False) # streaming can be removed if the dataset is not very large.
    
    
    train_dataset = dataset["train"]
    
    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
    
    train_dataset = train_dataset.map(tokenize_hotpot_example, batched=False, fn_kwargs={"model": model, "mem": MEM_TOKENS})

    # print(train_dataset[0].keys())  # 確保包含 'answer_ids'

    # eval_dataset = eval_dataset.map(contrastive_adversarial_tokenize_function, batched=False, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.
    
    # data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    
    train_model(model, train_dataset, None, training_args, custom_data_collator)

main()