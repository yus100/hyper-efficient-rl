#!/usr/bin/env python3
"""
Script for comprehensive evaluation of mathematical reasoning models.
Week 4 (Day 22-24) of the project timeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.logging import ResultsLogger, setup_logging
from utils.helpers import ensure_dir
from evaluation.evaluator import run_comprehensive_evaluation, compare_models
from evaluation.metrics import compute_all_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate mathematical reasoning models")
    parser.add_argument("--model-paths", type=str, nargs="+", required=True,
                       help="Paths to model checkpoints to evaluate")
    parser.add_argument("--model-names", type=str, nargs="*",
                       help="Names for models (for comparison reports)")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for evaluation results")
    parser.add_argument("--benchmarks", type=str, nargs="*", 
                       choices=["gsm8k", "math", "svamp"],
                       help="Benchmarks to evaluate on (overrides config)")
    parser.add_argument("--baseline-results", type=str, default=None,
                       help="Path to baseline results for comparison")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save model predictions")
    parser.add_argument("--compare-models", action="store_true",
                       help="Generate model comparison report")
    parser.add_argument("--ablation-study", action="store_true",
                       help="Run ablation study analysis")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Override benchmarks if specified
    if args.benchmarks:
        config["evaluation"]["benchmarks"] = args.benchmarks
    
    # Override save predictions if specified
    if args.save_predictions:
        config["evaluation"]["save_predictions"] = True
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Initialize results logger
    results_logger = ResultsLogger(args.output_dir)
    
    try:
        if len(args.model_paths) == 1 and not args.compare_models:
            # Single model evaluation
            model_path = args.model_paths[0]
            model_name = args.model_names[0] if args.model_names else Path(model_path).name
            
            print(f"Evaluating model: {model_name}")
            
            # Load baseline results if provided
            baseline_results = None
            if args.baseline_results:
                import json
                with open(args.baseline_results, 'r') as f:
                    baseline_results = json.load(f)
            
            # Run comprehensive evaluation
            results = run_comprehensive_evaluation(
                model_path=model_path,
                config=config,
                baseline_results=baseline_results
            )
            
            # Log results
            results_logger.log_experiment_results(model_name, results)
            
            # Generate and save report
            report = results_logger.generate_results_report()
            with open(os.path.join(args.output_dir, f"{model_name}_evaluation_report.txt"), 'w') as f:
                f.write(report)
            
            print(f"Evaluation completed! Results saved to {args.output_dir}")
            
        else:
            # Multiple model comparison
            print(f"Comparing {len(args.model_paths)} models...")
            
            # Prepare model names
            if args.model_names:
                if len(args.model_names) != len(args.model_paths):
                    raise ValueError("Number of model names must match number of model paths")
                model_names = args.model_names
            else:
                model_names = [Path(path).name for path in args.model_paths]
            
            # Run comparison
            comparison_df = compare_models(args.model_paths, config)
            
            # Save comparison results
            comparison_df.to_csv(os.path.join(args.output_dir, "model_comparison.csv"))
            
            # Log comparison results
            comparison_dict = {name: row.to_dict() for name, row in comparison_df.iterrows()}
            results_logger.log_model_comparison(comparison_dict)
            
            print(f"Model comparison completed! Results saved to {args.output_dir}")
        
        # Run ablation study if requested
        if args.ablation_study:
            print("Running ablation study...")
            # This would be implemented based on specific ablation requirements
            # For now, just placeholder
            ablation_results = {"placeholder": "ablation study results"}
            results_logger.log_ablation_study(ablation_results)
        
        # Save all results
        results_logger.save_results("evaluation_results.json")
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
