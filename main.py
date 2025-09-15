#!/usr/bin/env python3
"""
Main entry point for the Hyper-Efficient RL Mathematical Reasoning project.
Provides unified interface for all project operations.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_data_preparation(args):
    """Run data preparation script."""
    cmd = [
        sys.executable, "scripts/prepare_data.py",
        "--config", args.config,
        "--output-dir", args.data_output_dir,
    ]
    
    if args.datasets:
        cmd.extend(["--datasets"] + args.datasets)
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.create_curriculum:
        cmd.append("--create-curriculum")
    if args.estimate_difficulty:
        cmd.append("--estimate-difficulty")
    if args.debug:
        cmd.append("--debug")
    
    subprocess.run(cmd, check=True)


def run_sft_training(args):
    """Run supervised fine-tuning script."""
    cmd = [
        sys.executable, "scripts/train_sft.py",
        "--config", args.sft_config,
        "--output-dir", args.sft_output_dir,
    ]
    
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.resume_sft_checkpoint:
        cmd.extend(["--resume-from-checkpoint", args.resume_sft_checkpoint])
    if args.debug:
        cmd.append("--debug")
    
    subprocess.run(cmd, check=True)


def run_rl_training(args):
    """Run reinforcement learning training script."""
    cmd = [
        sys.executable, "scripts/train_rl.py",
        "--config", args.rl_config,
        "--sft-model-path", args.sft_model_path,
        "--output-dir", args.rl_output_dir,
    ]
    
    if args.resume_rl_checkpoint:
        cmd.extend(["--resume-from-checkpoint", args.resume_rl_checkpoint])
    if not args.disable_curriculum:
        cmd.append("--curriculum-enabled")
    if not args.disable_length_reward:
        cmd.append("--length-aware-reward")
    if args.debug:
        cmd.append("--debug")
    
    subprocess.run(cmd, check=True)


def run_evaluation(args):
    """Run evaluation script."""
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model-paths"] + args.model_paths + [
        "--config", args.config,
        "--output-dir", args.eval_output_dir,
    ]
    
    if args.model_names:
        cmd.extend(["--model-names"] + args.model_names)
    if args.benchmarks:
        cmd.extend(["--benchmarks"] + args.benchmarks)
    if args.baseline_results:
        cmd.extend(["--baseline-results", args.baseline_results])
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.compare_models:
        cmd.append("--compare-models")
    if args.ablation_study:
        cmd.append("--ablation-study")
    if args.debug:
        cmd.append("--debug")
    
    subprocess.run(cmd, check=True)


def run_full_pipeline(args):
    """Run the complete training pipeline."""
    print("Starting full pipeline...")
    
    # Step 1: Data preparation
    print("\n=== Step 1: Data Preparation ===")
    run_data_preparation(args)
    
    # Step 2: SFT training
    print("\n=== Step 2: Supervised Fine-Tuning ===")
    run_sft_training(args)
    
    # Step 3: RL training
    print("\n=== Step 3: Reinforcement Learning Training ===")
    # Update SFT model path for RL training
    args.sft_model_path = args.sft_output_dir
    run_rl_training(args)
    
    # Step 4: Evaluation
    print("\n=== Step 4: Evaluation ===")
    # Set up model paths for comparison
    args.model_paths = [args.sft_output_dir, args.rl_output_dir]
    args.model_names = ["SFT-Baseline", "RL-Final"]
    args.compare_models = True
    run_evaluation(args)
    
    print("\n=== Pipeline Complete! ===")
    print(f"SFT Model: {args.sft_output_dir}")
    print(f"RL Model: {args.rl_output_dir}")
    print(f"Results: {args.eval_output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyper-Efficient RL for Mathematical Reasoning")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Global arguments
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Base configuration file")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Data preparation subcommand
    data_parser = subparsers.add_parser("prepare-data", help="Prepare and preprocess datasets")
    data_parser.add_argument("--data-output-dir", type=str, default="data/processed",
                           help="Output directory for processed datasets")
    data_parser.add_argument("--datasets", type=str, nargs="*",
                           choices=["gsm8k", "math", "svamp"],
                           help="Datasets to prepare")
    data_parser.add_argument("--max-samples", type=int,
                           help="Maximum samples per dataset")
    data_parser.add_argument("--create-curriculum", action="store_true",
                           help="Create curriculum learning batches")
    data_parser.add_argument("--estimate-difficulty", action="store_true",
                           help="Estimate problem difficulty")
    
    # SFT training subcommand
    sft_parser = subparsers.add_parser("train-sft", help="Run supervised fine-tuning")
    sft_parser.add_argument("--sft-config", type=str, default="config/sft_config.yaml",
                          help="SFT configuration file")
    sft_parser.add_argument("--model-name", type=str,
                          help="Model name or path")
    sft_parser.add_argument("--sft-output-dir", type=str, default="checkpoints/sft",
                          help="Output directory for SFT model")
    sft_parser.add_argument("--resume-sft-checkpoint", type=str,
                          help="Resume SFT from checkpoint")
    
    # RL training subcommand
    rl_parser = subparsers.add_parser("train-rl", help="Run reinforcement learning training")
    rl_parser.add_argument("--rl-config", type=str, default="config/rl_config.yaml",
                         help="RL configuration file")
    rl_parser.add_argument("--sft-model-path", type=str, required=True,
                         help="Path to SFT model")
    rl_parser.add_argument("--rl-output-dir", type=str, default="checkpoints/rl",
                         help="Output directory for RL model")
    rl_parser.add_argument("--resume-rl-checkpoint", type=str,
                         help="Resume RL from checkpoint")
    rl_parser.add_argument("--disable-curriculum", action="store_true",
                         help="Disable curriculum learning")
    rl_parser.add_argument("--disable-length-reward", action="store_true",
                         help="Disable length-aware reward")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--model-paths", type=str, nargs="+", required=True,
                           help="Paths to models to evaluate")
    eval_parser.add_argument("--model-names", type=str, nargs="*",
                           help="Names for models")
    eval_parser.add_argument("--eval-output-dir", type=str, default="results",
                           help="Output directory for results")
    eval_parser.add_argument("--benchmarks", type=str, nargs="*",
                           choices=["gsm8k", "math", "svamp"],
                           help="Benchmarks to evaluate on")
    eval_parser.add_argument("--baseline-results", type=str,
                           help="Path to baseline results")
    eval_parser.add_argument("--save-predictions", action="store_true",
                           help="Save model predictions")
    eval_parser.add_argument("--compare-models", action="store_true",
                           help="Generate model comparison")
    eval_parser.add_argument("--ablation-study", action="store_true",
                           help="Run ablation study")
    
    # Full pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete training pipeline")
    pipeline_parser.add_argument("--model-name", type=str, 
                                default="microsoft/DialoGPT-medium",
                                help="Base model name")
    pipeline_parser.add_argument("--sft-config", type=str, default="config/sft_config.yaml",
                                help="SFT configuration file")
    pipeline_parser.add_argument("--rl-config", type=str, default="config/rl_config.yaml",
                                help="RL configuration file")
    pipeline_parser.add_argument("--data-output-dir", type=str, default="data/processed",
                                help="Data output directory")
    pipeline_parser.add_argument("--sft-output-dir", type=str, default="checkpoints/sft",
                                help="SFT output directory")
    pipeline_parser.add_argument("--rl-output-dir", type=str, default="checkpoints/rl",
                                help="RL output directory")
    pipeline_parser.add_argument("--eval-output-dir", type=str, default="results",
                                help="Evaluation output directory")
    pipeline_parser.add_argument("--datasets", type=str, nargs="*", 
                                default=["gsm8k", "math"],
                                choices=["gsm8k", "math", "svamp"],
                                help="Datasets to use")
    pipeline_parser.add_argument("--benchmarks", type=str, nargs="*",
                                default=["gsm8k", "math", "svamp"],
                                choices=["gsm8k", "math", "svamp"],
                                help="Benchmarks for evaluation")
    pipeline_parser.add_argument("--create-curriculum", action="store_true", default=True,
                                help="Create curriculum learning batches")
    pipeline_parser.add_argument("--estimate-difficulty", action="store_true", default=True,
                                help="Estimate problem difficulty")
    pipeline_parser.add_argument("--disable-curriculum", action="store_true",
                                help="Disable curriculum learning in RL")
    pipeline_parser.add_argument("--disable-length-reward", action="store_true",
                                help="Disable length-aware reward in RL")
    pipeline_parser.add_argument("--save-predictions", action="store_true", default=True,
                                help="Save model predictions")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command is None:
        print("No command specified. Use --help for available commands.")
        sys.exit(1)
    
    try:
        if args.command == "prepare-data":
            run_data_preparation(args)
        elif args.command == "train-sft":
            run_sft_training(args)
        elif args.command == "train-rl":
            run_rl_training(args)
        elif args.command == "evaluate":
            run_evaluation(args)
        elif args.command == "pipeline":
            run_full_pipeline(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
