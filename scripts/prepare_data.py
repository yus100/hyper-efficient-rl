#!/usr/bin/env python3
"""
Script for data preparation and preprocessing.
Week 1 (Day 1-2) of the project timeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.logging import setup_logging
from utils.helpers import set_seed, ensure_dir
from data.dataset_loader import MathDatasetLoader
from data.preprocessing import DifficultyEstimator, create_curriculum_batches


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare and preprocess mathematical reasoning datasets")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed datasets")
    parser.add_argument("--datasets", type=str, nargs="*", 
                       choices=["gsm8k", "math", "svamp"],
                       help="Datasets to prepare (overrides config)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (overrides config)")
    parser.add_argument("--create-curriculum", action="store_true",
                       help="Create curriculum learning batches")
    parser.add_argument("--estimate-difficulty", action="store_true",
                       help="Estimate problem difficulty")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    """Main data preparation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Ensure data section exists
    if "data" not in config:
        config["data"] = {}
    
    # Override config with command line arguments
    if args.datasets:
        config["data"]["datasets"] = args.datasets
    if args.max_samples:
        config["data"]["max_samples"] = args.max_samples
    
    # Set default values if not present
    if "datasets" not in config["data"]:
        config["data"]["datasets"] = ["gsm8k", "math"]
    if "max_samples" not in config["data"]:
        config["data"]["max_samples"] = 10000
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    try:
        # Initialize dataset loader
        dataset_loader = MathDatasetLoader(config)
        
        # Process each dataset
        for dataset_name in config["data"]["datasets"]:
            print(f"Processing {dataset_name} dataset...")
            
            # Load raw dataset
            train_dataset = dataset_loader.load_dataset(dataset_name, "train")
            test_dataset = dataset_loader.load_dataset(dataset_name, "test")
            
            # Preprocess datasets
            if dataset_name == "gsm8k":
                train_dataset = dataset_loader.preprocess_gsm8k(train_dataset)
                test_dataset = dataset_loader.preprocess_gsm8k(test_dataset)
            elif dataset_name == "math":
                train_dataset = dataset_loader.preprocess_math(train_dataset)
                test_dataset = dataset_loader.preprocess_math(test_dataset)
            elif dataset_name == "svamp":
                test_dataset = dataset_loader.preprocess_svamp(test_dataset)
            
            # Estimate difficulty if requested
            if args.estimate_difficulty:
                print(f"Estimating difficulty for {dataset_name}...")
                difficulty_estimator = DifficultyEstimator()
                
                # Add difficulty scores to datasets
                def add_difficulty_score(example):
                    difficulty = difficulty_estimator.estimate_heuristic_difficulty(
                        example["problem"], 
                        example.get("solution", "")
                    )
                    example["difficulty"] = difficulty
                    return example
                
                train_dataset = train_dataset.map(add_difficulty_score)
                if dataset_name != "svamp":  # SVAMP is evaluation only
                    test_dataset = test_dataset.map(add_difficulty_score)
            
            # Create curriculum batches if requested
            if args.create_curriculum and dataset_name != "svamp":
                print(f"Creating curriculum batches for {dataset_name}...")
                difficulty_estimator = DifficultyEstimator()
                curriculum_batches = create_curriculum_batches(
                    train_dataset, 
                    difficulty_estimator,
                    num_levels=5
                )
                
                # Save curriculum batches
                for level, batch in enumerate(curriculum_batches):
                    batch_output_path = os.path.join(args.output_dir, f"{dataset_name}_curriculum_level_{level}")
                    batch.save_to_disk(batch_output_path)
                    print(f"Saved curriculum level {level} to {batch_output_path}")
            
            # Save processed datasets
            train_output_path = os.path.join(args.output_dir, f"{dataset_name}_train")
            test_output_path = os.path.join(args.output_dir, f"{dataset_name}_test")
            
            train_dataset.save_to_disk(train_output_path)
            test_dataset.save_to_disk(test_output_path)
            
            print(f"Saved processed {dataset_name} datasets:")
            print(f"  Train: {train_output_path}")
            print(f"  Test: {test_output_path}")
        
        print(f"Data preparation completed! Processed datasets saved to {args.output_dir}")
        
        # Generate data summary
        summary_path = os.path.join(args.output_dir, "data_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Dataset Preparation Summary\n")
            f.write("==========================\n\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Output directory: {args.output_dir}\n")
            f.write(f"Datasets processed: {config['data']['datasets']}\n")
            f.write(f"Max samples per dataset: {config['data']['max_samples']}\n")
            f.write(f"Difficulty estimation: {args.estimate_difficulty}\n")
            f.write(f"Curriculum creation: {args.create_curriculum}\n")
            f.write(f"Random seed: {args.seed}\n")
        
        print(f"Data summary saved to {summary_path}")
        
    except Exception as e:
        print(f"Data preparation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
