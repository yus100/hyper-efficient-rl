#!/usr/bin/env python3
"""
Script for reinforcement learning training with curriculum learning and length-aware rewards.
Week 2-3 (Day 8-21) of the project timeline.
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
from training.rl_trainer import run_rl_training


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL model for mathematical reasoning")
    parser.add_argument("--config", type=str, default="config/rl_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--sft-model-path", type=str, required=True,
                       help="Path to SFT model checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rl",
                       help="Output directory for model checkpoints")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing datasets")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--curriculum-enabled", action="store_true", default=True,
                       help="Enable curriculum learning (SPEED)")
    parser.add_argument("--length-aware-reward", action="store_true", default=True,
                       help="Enable length-aware reward function")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    """Main RL training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_training_config("rl")
    
    # Override config with command line arguments
    config["curriculum"]["enabled"] = args.curriculum_enabled
    config["reward"]["enabled"] = args.length_aware_reward
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Initialize experiment logger
    logger = ExperimentLogger(config)
    experiment_name = f"rl_training_{'curriculum_' if args.curriculum_enabled else ''}{'length_aware' if args.length_aware_reward else 'standard'}"
    logger.init_wandb(
        project_name=config.get("logging", {}).get("project_name", "hyper-efficient-rl-math"),
        run_name=experiment_name,
        tags=["rl", "ppo", "mathematical-reasoning"] + 
             (["curriculum"] if args.curriculum_enabled else []) +
             (["length-aware"] if args.length_aware_reward else [])
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    try:
        # Load and prepare datasets
        print("Loading and preparing datasets...")
        datasets = load_and_prepare_datasets(config, tokenizer=None)  # tokenizer loaded in function
        
        # Run RL training
        print("Starting reinforcement learning training...")
        run_rl_training(
            config=config,
            sft_model_path=args.sft_model_path,
            dataset=datasets["train"],
            eval_dataset=datasets.get("eval"),
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        print(f"RL training completed! Model saved to {args.output_dir}")
        
    except Exception as e:
        print(f"RL training failed with error: {e}")
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
