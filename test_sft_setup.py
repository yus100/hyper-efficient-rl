#!/usr/bin/env python3
"""
Test script for SFT trainer setup and model loading.
Tests the implementation without running full training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.sft_trainer import MathSFTTrainer, formatting_prompts_func
from utils.config import ConfigManager
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    config_manager = ConfigManager()
    config = config_manager.load_training_config("sft")
    
    logger.info(f"Model name: {config.get('model', {}).get('name')}")
    logger.info(f"Max length: {config.get('model', {}).get('max_length')}")
    logger.info(f"Batch size: {config.get('training', {}).get('batch_size')}")
    logger.info(f"LoRA r: {config.get('model', {}).get('lora_config', {}).get('r')}")
    
    logger.info("✓ Configuration loading tests passed")
    return config


def test_formatting_function():
    """Test the prompt formatting function."""
    logger.info("Testing prompt formatting...")
    
    example = {
        "problem": "What is 15 + 27?",
        "solution": "15 + 27 = 42"
    }
    
    formatted = formatting_prompts_func(example)
    logger.info(f"Formatted example:\n{formatted}")
    
    # Test without solution
    example_no_solution = {
        "problem": "What is 20 × 5?",
        "solution": ""
    }
    
    formatted_no_sol = formatting_prompts_func(example_no_solution)
    logger.info(f"Formatted (no solution):\n{formatted_no_sol}")
    
    logger.info("✓ Formatting function tests passed")


def test_trainer_initialization():
    """Test SFT trainer initialization."""
    logger.info("Testing SFT trainer initialization...")
    
    # Create minimal config
    config = {
        "model": {
            "name": "Qwen/Qwen2.5-4B",
            "max_length": 2048,
            "use_peft": True,
            "lora_config": {
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "num_epochs": 1,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "evaluation_steps": 100,
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 2
        },
        "sft": {
            "max_seq_length": 1024,
            "packing": False
        },
        "logging": {
            "use_wandb": False
        }
    }
    
    # Initialize trainer
    trainer = MathSFTTrainer(config)
    logger.info(f"Trainer initialized with device: {trainer.device}")
    
    logger.info("✓ Trainer initialization tests passed")
    return trainer


def test_synthetic_dataset():
    """Test with synthetic dataset."""
    logger.info("Testing with synthetic dataset...")
    
    # Create synthetic dataset
    synthetic_data = {
        "problem": [
            "What is 2 + 3?",
            "Calculate 15 × 4",
            "Solve for x: x + 5 = 12",
            "Find the area of a square with side length 6",
            "What is 50% of 80?"
        ],
        "solution": [
            "2 + 3 = 5",
            "15 × 4 = 60",
            "x + 5 = 12\nx = 12 - 5\nx = 7",
            "Area = side × side = 6 × 6 = 36",
            "50% of 80 = 0.5 × 80 = 40"
        ]
    }
    
    dataset = Dataset.from_dict(synthetic_data)
    logger.info(f"Created synthetic dataset with {len(dataset)} examples")
    
    # Test formatting
    for i, example in enumerate(dataset.select(range(2))):
        formatted = formatting_prompts_func(example)
        logger.info(f"Example {i+1} formatted length: {len(formatted)} characters")
    
    logger.info("✓ Synthetic dataset tests passed")
    return dataset


def main():
    """Run all tests."""
    logger.info("Starting SFT setup tests...")
    
    try:
        # Test configuration
        config = test_config_loading()
        
        # Test formatting
        test_formatting_function()
        
        # Test trainer initialization
        trainer = test_trainer_initialization()
        
        # Test synthetic dataset
        dataset = test_synthetic_dataset()
        
        logger.info("🎉 All SFT setup tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. Install additional dependencies: pip install trl peft bitsandbytes accelerate")
        logger.info("2. Run model loading test (requires GPU and internet)")
        logger.info("3. Start SFT training with: python scripts/train_sft.py")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
