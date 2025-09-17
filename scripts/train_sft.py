#!/usr/bin/env python3
"""
Script for supervised fine-tuning (SFT) of mathematical reasoning model.
Week 1 (Day 3-5) of the project timeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.logging import ExperimentLogger, setup_logging
from utils.helpers import set_seed, ensure_dir
from data.dataset_loader import load_and_prepare_datasets
from training.sft_trainer import run_sft_training


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SFT model for mathematical reasoning")
    parser.add_argument("--config", type=str, default="config/sft_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name or path (overrides config)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft",
                       help="Output directory for model checkpoints")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing datasets")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_training_config("sft")
    
    # Override config with command line arguments
    if args.model_name:
        config["model"]["name"] = args.model_name
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Initialize experiment logger
    logger = ExperimentLogger(config)
    experiment_name = f"sft_training_{config['model']['name'].split('/')[-1]}"
    logger.init_wandb(
        project_name=config.get("logging", {}).get("project_name", "hyper-efficient-rl-math"),
        run_name=experiment_name,
        tags=["sft", "mathematical-reasoning"]
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    try:
        # Load and prepare datasets
        print("Loading and preparing datasets...")
        datasets = load_and_prepare_datasets(config, tokenizer=None)  # tokenizer loaded in function
        
        # Format datasets for SFT if needed
        print("Formatting datasets for SFT training...")
        
        # Run SFT training
        print("Starting supervised fine-tuning...")
        result = run_sft_training(
            config=config,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("eval"),
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        print("Training results:")
        print(f"- Training loss: {result['train_result'].training_loss:.4f}")
        if result['eval_result']:
            print(f"- Evaluation loss: {result['eval_result']['eval_loss']:.4f}")
        
        print("\nSample generations:")
        for i, sample in enumerate(result['samples'][:3]):
            print(f"\nSample {i+1}:")
            print(f"Problem: {sample['problem']}")
            print(f"Generated: {sample['generated_solution'][:200]}...")  # Truncate for display
        
        print(f"Training completed! Model saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
