"""
Supervised Fine-Tuning (SFT) trainer for mathematical reasoning.
Uses parameter-efficient fine-tuning with LoRA.
"""

import os
import torch
import logging
from typing import Dict, Optional, List
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MathSFTTrainer:
    """
    Supervised fine-tuning trainer for mathematical reasoning tasks.
    Implements parameter-efficient fine-tuning using LoRA.
    """
    
    def __init__(self, config: Dict):
        """Initialize the SFT trainer with configuration."""
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.sft_config = config.get("sft", {})
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized MathSFTTrainer with device: {self.device}")
    
    def setup_model_and_tokenizer(self, model_name: str = None):
        """Load and setup the base model and tokenizer."""
        model_name = model_name or self.model_config.get("name", "Qwen/Qwen2.5-4B")
        logger.info(f"Loading model: {model_name}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
        
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration for parameter-efficient fine-tuning."""
        lora_config = self.model_config.get("lora_config", {})
        
        config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
        return config
    
    def prepare_model_for_training(self):
        """Prepare model for training with LoRA and other optimizations."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model_and_tokenizer first.")
        
        # Apply LoRA
        if self.model_config.get("use_peft", True):
            lora_config = self.setup_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            logger.info(
                f"Trainable params: {trainable_params:,} || "
                f"All params: {all_param:,} || "
                f"Trainable%: {100 * trainable_params / all_param:.2f}"
            )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create training arguments for the SFT process."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.training_config.get("batch_size", 4),
            per_device_eval_batch_size=self.training_config.get("batch_size", 4),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.training_config.get("learning_rate", 5e-5),
            num_train_epochs=self.training_config.get("num_epochs", 3),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            
            # Optimization settings
            warmup_ratio=self.training_config.get("warmup_ratio", 0.1),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            
            # Memory and performance
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=True,
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            
            # Logging and saving
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 500),
            eval_steps=self.training_config.get("evaluation_steps", 500),
            evaluation_strategy="steps" if self.training_config.get("evaluation_steps") else "no",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Reporting
            report_to=["wandb"] if self.config.get("logging", {}).get("use_wandb", False) else [],
            run_name=f"sft_math_reasoning_{self.model_config.get('name', 'qwen').split('/')[-1]}",
            
            # Other settings
            remove_unused_columns=False,
            push_to_hub=False,
            hub_model_id=None,
        )
        
        return training_args
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, output_dir: str = "checkpoints/sft"):
        """Setup the SFT trainer with datasets."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir)
        
        # Setup trainer
        self.trainer = CustomSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_prompts_func,
            max_seq_length=self.sft_config.get("max_seq_length", 2048),
            packing=self.sft_config.get("packing", False),
        )
        
        logger.info("SFT Trainer setup complete")
    
    def train(self):
        """Execute the supervised fine-tuning process."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer first.")
        
        logger.info("Starting SFT training...")
        
        # Train the model
        train_result = self.trainer.train()
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model and tokenizer."""
        if self.trainer is None:
            raise ValueError("Trainer not available for saving.")
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save the model
        self.trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        if hasattr(self.trainer, 'state'):
            self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        logger.info("Model saved successfully")
    
    def evaluate(self, eval_dataset: Dataset = None) -> Dict[str, float]:
        """Evaluate the model on evaluation dataset."""
        if self.trainer is None:
            raise ValueError("Trainer not setup.")
        
        logger.info("Running evaluation...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"Evaluation results: {eval_result}")
        return eval_result
    
    def generate_sample_outputs(self, test_problems: List[str], num_samples: int = 5):
        """Generate sample outputs for manual inspection."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        logger.info(f"Generating {num_samples} sample outputs...")
        
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for i, problem in enumerate(test_problems[:num_samples]):
                # Format the problem
                prompt = formatting_prompts_func({"problem": problem, "solution": ""}).split("Solution:")[0] + "Solution:"
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode
                response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                
                samples.append({
                    "problem": problem,
                    "generated_solution": response.strip()
                })
                
                logger.info(f"Sample {i+1}:")
                logger.info(f"Problem: {problem}")
                logger.info(f"Generated: {response.strip()}")
                logger.info("-" * 50)
        
        return samples


class CustomSFTTrainer(SFTTrainer):
    """
    Custom SFT trainer with mathematical reasoning specific features.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize custom SFT trainer."""
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation for mathematical reasoning."""
        # Use the standard causal language modeling loss
        return super().compute_loss(model, inputs, return_outputs)
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with mathematical reasoning specific metrics."""
        # Add custom metrics if needed
        if "train_loss" in logs:
            logs["perplexity"] = np.exp(logs["train_loss"])
        
        super().log(logs)


def formatting_prompts_func(example: Dict) -> str:
    """
    Format examples for SFT training.
    
    Args:
        example: Single example from dataset
        
    Returns:
        Formatted prompt string
    """
    instruction = "Solve the following mathematical problem step by step:"
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    
    # Create the full training example
    if solution:
        text = f"{instruction}\n\nProblem: {problem}\n\nSolution: {solution}"
    else:
        text = f"{instruction}\n\nProblem: {problem}\n\nSolution:"
    
    return text


def run_sft_training(config: Dict, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, 
                     output_dir: str = "checkpoints/sft", resume_from_checkpoint: str = None):
    """
    Main function to run supervised fine-tuning.
    
    Args:
        config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        output_dir: Output directory for model checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    logger.info("Starting SFT training pipeline...")
    
    # Initialize trainer
    sft_trainer = MathSFTTrainer(config)
    
    # Setup model and tokenizer
    sft_trainer.setup_model_and_tokenizer()
    
    # Prepare model for training
    sft_trainer.prepare_model_for_training()
    
    # Format datasets for SFT
    logger.info("Formatting datasets for SFT...")
    # The datasets should already be formatted by the dataset loader
    
    # Setup trainer
    sft_trainer.setup_trainer(train_dataset, eval_dataset, output_dir)
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        logger.info(f"Resuming training from {resume_from_checkpoint}")
    
    # Train the model
    train_result = sft_trainer.train()
    
    # Save the model
    sft_trainer.save_model(output_dir)
    
    # Run evaluation if eval dataset provided
    if eval_dataset is not None:
        eval_result = sft_trainer.evaluate(eval_dataset)
        logger.info(f"Final evaluation results: {eval_result}")
    
    # Generate sample outputs
    test_problems = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "Solve for x: 2x + 7 = 19"
    ]
    
    samples = sft_trainer.generate_sample_outputs(test_problems)
    
    logger.info("SFT training completed successfully!")
    
    return {
        "trainer": sft_trainer,
        "train_result": train_result,
        "eval_result": eval_result if eval_dataset else None,
        "samples": samples
    }
