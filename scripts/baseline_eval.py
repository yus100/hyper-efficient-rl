#!/usr/bin/env python3
"""
Baseline evaluation script for raw Qwen2.5 4B model.
Tests the model's mathematical reasoning capabilities before fine-tuning.
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.logging import setup_logging
from utils.helpers import set_seed, ensure_dir
from data.dataset_loader import load_and_prepare_datasets
from data.preprocessing import MathTextProcessor
from evaluation.metrics import AccuracyMetrics

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """
    Evaluator for baseline model performance on mathematical reasoning.
    """
    
    def __init__(self, config: Dict):
        """Initialize baseline evaluator."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = MathTextProcessor()
        self.device = None
        
    def setup_model(self, model_name: str = None):
        """Load the baseline model and tokenizer."""
        model_name = model_name or self.config.get("model", {}).get("name", "Qwen/Qwen2.5-4B")
        logger.info(f"Loading baseline model: {model_name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Setup tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Setup quantization for memory efficiency
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
            )
            
            self.model.eval()
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Please install: pip install transformers torch bitsandbytes accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, problem: str, max_new_tokens: int = 512) -> str:
        """Generate response for a mathematical problem."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model first.")
        
        # Format the prompt
        prompt = f"Solve the following mathematical problem step by step:\n\nProblem: {problem}\n\nSolution:"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed for problem: {problem[:50]}... Error: {e}")
            return ""
    
    def evaluate_dataset(self, dataset, dataset_name: str, max_samples: int = None) -> Dict:
        """Evaluate model on a dataset."""
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        results = {
            "dataset": dataset_name,
            "total_samples": len(dataset),
            "predictions": [],
            "ground_truths": [],
            "problems": [],
            "response_times": [],
            "token_counts": []
        }
        
        correct_predictions = 0
        total_time = 0
        
        for i, example in enumerate(dataset):
            problem = example["problem"]
            ground_truth = example["numerical_answer"]
            
            logger.info(f"Evaluating sample {i+1}/{len(dataset)}: {problem[:50]}...")
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(problem)
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            # Extract predicted answer
            predicted_answer = self.processor.extract_numerical_answer(response)
            predicted_str = str(predicted_answer) if predicted_answer is not None else ""
            
            # Count tokens
            token_count = len(self.tokenizer.encode(response)) if response else 0
            
            # Check correctness
            is_correct = self.compare_answers(predicted_str, str(ground_truth))
            if is_correct:
                correct_predictions += 1
            
            # Store results
            results["predictions"].append(predicted_str)
            results["ground_truths"].append(str(ground_truth))
            results["problems"].append(problem)
            results["response_times"].append(response_time)
            results["token_counts"].append(token_count)
            
            logger.info(f"  Ground truth: {ground_truth}")
            logger.info(f"  Predicted: {predicted_str}")
            logger.info(f"  Correct: {is_correct}")
            logger.info(f"  Response time: {response_time:.2f}s")
            logger.info(f"  Generated tokens: {token_count}")
            logger.info("-" * 50)
        
        # Compute final metrics
        accuracy = correct_predictions / len(dataset)
        avg_response_time = total_time / len(dataset)
        avg_token_count = sum(results["token_counts"]) / len(dataset)
        
        results.update({
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "avg_response_time": avg_response_time,
            "total_time": total_time,
            "avg_token_count": avg_token_count,
            "tokens_per_second": avg_token_count / avg_response_time if avg_response_time > 0 else 0
        })
        
        logger.info(f"\n{dataset_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.3f} ({correct_predictions}/{len(dataset)})")
        logger.info(f"  Avg response time: {avg_response_time:.2f}s")
        logger.info(f"  Avg token count: {avg_token_count:.1f}")
        logger.info(f"  Tokens/second: {results['tokens_per_second']:.1f}")
        
        return results
    
    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth."""
        try:
            # Try numerical comparison
            pred_num = float(predicted.replace(",", "").strip())
            gt_num = float(ground_truth.replace(",", "").strip())
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            return predicted.strip().lower() == ground_truth.strip().lower()
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        ensure_dir(os.path.dirname(output_path))
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, all_results: List[Dict]) -> str:
        """Generate a comprehensive evaluation report."""
        report = ["Baseline Model Evaluation Report", "=" * 40, ""]
        
        model_name = self.config.get("model", {}).get("name", "Unknown")
        report.extend([
            f"Model: {model_name}",
            f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Device: {self.device}",
            ""
        ])
        
        # Summary table
        report.extend(["Dataset Performance Summary:", "-" * 30])
        for result in all_results:
            dataset = result["dataset"]
            accuracy = result["accuracy"]
            samples = result["total_samples"]
            avg_time = result["avg_response_time"]
            avg_tokens = result["avg_token_count"]
            
            report.append(f"{dataset:>10}: {accuracy:.3f} acc, {samples:>3} samples, {avg_time:.2f}s/sample, {avg_tokens:.0f} tokens")
        
        report.append("")
        
        # Detailed analysis
        for result in all_results:
            report.extend([
                f"Detailed Analysis - {result['dataset']}:",
                "-" * 25,
                f"  Total samples: {result['total_samples']}",
                f"  Correct predictions: {result['correct_predictions']}",
                f"  Accuracy: {result['accuracy']:.3f}",
                f"  Average response time: {result['avg_response_time']:.2f}s",
                f"  Total evaluation time: {result['total_time']:.1f}s",
                f"  Average tokens generated: {result['avg_token_count']:.1f}",
                f"  Generation speed: {result['tokens_per_second']:.1f} tokens/s",
                ""
            ])
        
        return "\n".join(report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baseline evaluation for mathematical reasoning")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name (overrides config)")
    parser.add_argument("--datasets", type=str, nargs="*", default=["gsm8k"],
                       choices=["gsm8k", "math", "svamp"],
                       help="Datasets to evaluate on")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum samples per dataset")
    parser.add_argument("--output-dir", type=str, default="results/baseline",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
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
    
    # Override model name if specified
    if args.model_name:
        if "model" not in config:
            config["model"] = {}
        config["model"]["name"] = args.model_name
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    logger.info("Starting baseline evaluation...")
    logger.info(f"Model: {config.get('model', {}).get('name')}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Max samples per dataset: {args.max_samples}")
    
    try:
        # Initialize evaluator
        evaluator = BaselineEvaluator(config)
        
        # Load model
        evaluator.setup_model()
        
        # Load datasets
        logger.info("Loading datasets...")
        datasets = load_and_prepare_datasets(config)
        
        # Run evaluation on each dataset
        all_results = []
        
        for dataset_name in args.datasets:
            if dataset_name == "svamp":
                dataset_key = "svamp_test"
            else:
                dataset_key = f"{dataset_name}_test"
            
            if dataset_key not in datasets:
                logger.warning(f"Dataset {dataset_key} not found, skipping...")
                continue
            
            dataset = datasets[dataset_key]
            result = evaluator.evaluate_dataset(dataset, dataset_name, args.max_samples)
            all_results.append(result)
            
            # Save individual results
            result_file = os.path.join(args.output_dir, f"baseline_{dataset_name}_results.json")
            evaluator.save_results(result, result_file)
        
        # Generate and save comprehensive report
        report = evaluator.generate_report(all_results)
        report_file = os.path.join(args.output_dir, "baseline_evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save combined results
        combined_results = {
            "model": config.get("model", {}).get("name"),
            "evaluation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "datasets": all_results,
            "summary": {
                "total_datasets": len(all_results),
                "overall_accuracy": sum(r["accuracy"] for r in all_results) / len(all_results),
                "total_samples": sum(r["total_samples"] for r in all_results),
                "total_time": sum(r["total_time"] for r in all_results)
            }
        }
        
        combined_file = os.path.join(args.output_dir, "baseline_combined_results.json")
        evaluator.save_results(combined_results, combined_file)
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE EVALUATION COMPLETE")
        print("="*50)
        print(report)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
