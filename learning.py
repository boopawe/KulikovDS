import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            padded = label + [-100] * (max_len - len(label))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch



MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  
DATA_PATH = "data.jsonl"                   
MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)


lora_config = LoraConfig(
    r=32,                       
    lora_alpha=64,              
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False


raw_data = load_dataset("json", data_files=DATA_PATH, split="train")

def format_row(example):
    prompt = example["prompt"]
    answer = example["response"]

    
    full_text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )

   
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

    input_ids = full_tokens["input_ids"]

   
    prompt_tokens = tokenizer(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    prompt_len = min(len(prompt_tokens["input_ids"]), len(input_ids))

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dataset = raw_data.map(format_row, remove_columns=raw_data.column_names)


training_args = TrainingArguments(
    output_dir="./my-lora-model-best",
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=4,
    num_train_epochs=10,             
    learning_rate=2e-4,              
    warmup_ratio=0.03,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)

data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./my-lora-model-best")
tokenizer.save_pretrained("./my-lora-model-best")
