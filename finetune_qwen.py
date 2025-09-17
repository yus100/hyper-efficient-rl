"""Fine-tune Qwen3-1.7B on GSM8K using Unsloth LoRA."""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from data_loader import load_gsm8k_data

def setup_model():
    """Load Qwen3-1.7B with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-1.7B-Base",
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer

def train_model():
    """Main training function."""
    print("Setting up model...")
    model, tokenizer = setup_model()
    
    print("Loading data...")
    train_dataset, _ = load_gsm8k_data()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=250,
        report_to=None,  # Disable wandb for simplicity
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("./qwen3-gsm8k-lora")
    
    # Save to HF format
    model.save_pretrained_merged("./qwen3-gsm8k-merged", tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    train_model()
    print("Training completed!")
