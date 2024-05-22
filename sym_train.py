from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.nn.parallel import DataParallel

from transformers import AutoModel, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=1024)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="datasets/tfshaman/gsm8k_sympy_v2",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=25, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.001, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=5, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    fsdp: str = field(default="auto_wrap", metadata={"help":"Use fsdp"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def formatting_func(example):
    text = f"### input: {example['input'][0]}\n### output: {example['output'][0]}\n### answer: {example['answer'][0]}\n### question: {example['question'][0]}\n### code_output: {example['code_output'][0]}\n"
    return text

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "tuned_model/gemma-it-qlora-mathinstruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

lora_config = LoraConfig(
    r=script_args.lora_r,
#    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj"],
    task_type="CAUSAL_LM",
    bias="none",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

model.add_adapter(lora_config, adapter_name="pysym_adapter")
model.disable_adapters()
model.set_adapter("pysym_adapter")

train_dataset = load_dataset(script_args.dataset_name, split="train")

# TODO: make that configurable
YOUR_HF_USERNAME = "xxx"
output_dir = f"{YOUR_HF_USERNAME}/gemma-mathinstruct-adaptor-sym"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    #fsdp=script_args.fsdp,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=lora_config,
    packing=script_args.packing,
    dataset_text_field="id",
    tokenizer=tokenizer,
    max_seq_length=script_args.max_seq_length,
    formatting_func=formatting_func,
)

trainer.train()

# save_dir = "tuned_model/gemma-mathinstruct-adapter-sym"
# model.save_pretrained(save_dir)