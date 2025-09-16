"""
Dataset loading and preprocessing for mathematical reasoning tasks.
Handles GSM8K, MATH, and SVAMP datasets.
"""

import os
import re
import json
import random
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import torch
import logging

logger = logging.getLogger(__name__)


class MathDatasetLoader:
    """
    Unified loader for mathematical reasoning datasets.
    Supports GSM8K, MATH, and SVAMP with consistent preprocessing.
    """
    
    def __init__(self, config: Dict):
        """Initialize the dataset loader with configuration."""
        self.config = config
        self.data_config = config.get("data", {})
        self.max_samples = self.data_config.get("max_samples", None)
        self.preprocessing_config = self.data_config.get("preprocessing", {})
        
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load a specific mathematical reasoning dataset."""
        logger.info(f"Loading {dataset_name} dataset, split: {split}")
        
        try:
            if dataset_name == "gsm8k":
                # Load GSM8K from HuggingFace datasets
                dataset = load_dataset("gsm8k", "main", split=split)
            elif dataset_name == "math":
                # Load MATH dataset from HuggingFace datasets
                dataset = load_dataset("hendrycks/competition_math", split=split)
            elif dataset_name == "svamp":
                # Load SVAMP dataset
                dataset = load_dataset("ChilleD/SVAMP", split="test")  # SVAMP only has test split
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            # Limit samples if specified
            if self.max_samples and len(dataset) > self.max_samples:
                # Use deterministic sampling for reproducibility
                indices = list(range(len(dataset)))
                random.Random(42).shuffle(indices)
                selected_indices = indices[:self.max_samples]
                dataset = dataset.select(selected_indices)
                logger.info(f"Limited {dataset_name} to {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            raise
    
    def preprocess_gsm8k(self, dataset: Dataset) -> Dataset:
        """Preprocess GSM8K dataset for training/evaluation."""
        logger.info("Preprocessing GSM8K dataset")
        
        def process_gsm8k_example(example):
            # GSM8K format: {"question": "...", "answer": "..."}
            problem = example["question"].strip()
            solution = example["answer"].strip()
            
            # Extract numerical answer from solution
            answer_match = re.search(r"#### ([\d,.-]+)", solution)
            numerical_answer = answer_match.group(1) if answer_match else ""
            
            # Clean numerical answer
            numerical_answer = numerical_answer.replace(",", "")
            
            # Create reasoning steps by splitting on newlines and filtering
            reasoning_steps = [step.strip() for step in solution.split("\n") if step.strip() and not step.startswith("####")]
            
            return {
                "problem": problem,
                "solution": solution,
                "numerical_answer": numerical_answer,
                "reasoning_steps": reasoning_steps,
                "dataset": "gsm8k",
                "difficulty": self.estimate_difficulty(problem, solution)
            }
        
        return dataset.map(process_gsm8k_example, remove_columns=dataset.column_names)
    
    def preprocess_math(self, dataset: Dataset) -> Dataset:
        """Preprocess MATH dataset for training/evaluation."""
        logger.info("Preprocessing MATH dataset")
        
        def process_math_example(example):
            # MATH format: {"problem": "...", "solution": "...", "level": "...", "type": "..."}
            problem = example["problem"].strip()
            solution = example["solution"].strip()
            level = example.get("level", "unknown")
            problem_type = example.get("type", "unknown")
            
            # Extract numerical answer (MATH uses different format)
            # Look for boxed answers like \boxed{answer}
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
            if boxed_match:
                numerical_answer = boxed_match.group(1)
            else:
                # Fallback: look for final numerical values
                number_matches = re.findall(r"[-+]?\d*\.?\d+", solution)
                numerical_answer = number_matches[-1] if number_matches else ""
            
            # Create reasoning steps
            reasoning_steps = [step.strip() for step in solution.split("\n") if step.strip()]
            
            return {
                "problem": problem,
                "solution": solution,
                "numerical_answer": numerical_answer,
                "reasoning_steps": reasoning_steps,
                "level": level,
                "type": problem_type,
                "dataset": "math",
                "difficulty": self.estimate_difficulty(problem, solution)
            }
        
        return dataset.map(process_math_example, remove_columns=dataset.column_names)
    
    def preprocess_svamp(self, dataset: Dataset) -> Dataset:
        """Preprocess SVAMP dataset for evaluation."""
        logger.info("Preprocessing SVAMP dataset")
        
        def process_svamp_example(example):
            # SVAMP format varies, adapt based on actual structure
            if "Body" in example and "Question" in example:
                problem = f"{example['Body']} {example['Question']}".strip()
            else:
                problem = example.get("question", example.get("problem", "")).strip()
            
            # SVAMP typically has numerical answers
            numerical_answer = str(example.get("Answer", example.get("answer", "")))
            
            # SVAMP doesn't usually have step-by-step solutions
            solution = f"The answer is {numerical_answer}"
            reasoning_steps = [solution]
            
            return {
                "problem": problem,
                "solution": solution,
                "numerical_answer": numerical_answer,
                "reasoning_steps": reasoning_steps,
                "dataset": "svamp",
                "difficulty": self.estimate_difficulty(problem, solution)
            }
        
        return dataset.map(process_svamp_example, remove_columns=dataset.column_names)
    
    def format_for_sft(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
        """Format dataset for supervised fine-tuning."""
        logger.info("Formatting dataset for SFT")
        
        def format_sft_example(example):
            # Create instruction-following format
            instruction = "Solve the following mathematical problem step by step:"
            problem = example["problem"]
            solution = example["solution"]
            
            # Format as conversation
            prompt = f"{instruction}\n\nProblem: {problem}\n\nSolution:"
            full_text = f"{prompt} {solution}"
            
            # Tokenize
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.get("model", {}).get("max_length", 512),
                padding=False,
                return_tensors=None
            )
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "text": full_text,
                "prompt": prompt,
                "labels": tokenized["input_ids"].copy(),  # For causal LM, labels = input_ids
                **{k: v for k, v in example.items() if k not in ["input_ids", "attention_mask", "text", "prompt", "labels"]}
            }
        
        return dataset.map(format_sft_example)
    
    def format_for_rl(self, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
        """Format dataset for reinforcement learning."""
        logger.info("Formatting dataset for RL")
        
        def format_rl_example(example):
            # For RL, we only need the problem as query
            instruction = "Solve the following mathematical problem step by step:"
            problem = example["problem"]
            query = f"{instruction}\n\nProblem: {problem}\n\nSolution:"
            
            # Tokenize query
            tokenized = tokenizer(
                query,
                truncation=True,
                max_length=self.config.get("model", {}).get("max_length", 512) // 2,  # Leave room for response
                padding=False,
                return_tensors=None
            )
            
            return {
                "query": query,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "ground_truth": example["solution"],
                "numerical_answer": example["numerical_answer"],
                **{k: v for k, v in example.items() if k not in ["query", "input_ids", "attention_mask", "ground_truth"]}
            }
        
        return dataset.map(format_rl_example)
    
    def filter_by_difficulty(self, dataset: Dataset, min_difficulty: float, max_difficulty: float) -> Dataset:
        """Filter dataset by estimated difficulty level."""
        def filter_func(example):
            difficulty = example.get("difficulty", 0.5)
            return min_difficulty <= difficulty <= max_difficulty
        
        filtered = dataset.filter(filter_func)
        logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered)} examples (difficulty: {min_difficulty}-{max_difficulty})")
        return filtered
    
    def estimate_difficulty(self, problem: str, solution: str = "") -> float:
        """Estimate the difficulty of a mathematical problem using heuristics."""
        difficulty_score = 0.0
        
        # Problem length factor (longer problems tend to be harder)
        problem_length = len(problem.split())
        difficulty_score += min(problem_length / 100, 0.3)
        
        # Solution length factor (longer solutions tend to be harder)
        if solution:
            solution_length = len(solution.split())
            difficulty_score += min(solution_length / 200, 0.3)
        
        # Mathematical complexity indicators
        math_keywords = [
            "algebra", "equation", "polynomial", "quadratic", "derivative", "integral",
            "geometry", "triangle", "circle", "probability", "statistics", "matrix",
            "logarithm", "exponential", "trigonometry", "calculus", "theorem"
        ]
        
        problem_lower = problem.lower()
        math_complexity = sum(1 for keyword in math_keywords if keyword in problem_lower)
        difficulty_score += min(math_complexity * 0.1, 0.2)
        
        # Number complexity (multiple numbers, decimals, fractions)
        numbers = re.findall(r"\d+\.?\d*", problem)
        difficulty_score += min(len(numbers) * 0.05, 0.2)
        
        # Ensure score is between 0 and 1
        return min(max(difficulty_score, 0.1), 1.0)
    
    def create_train_eval_split(self, dataset: Dataset, eval_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
        """Create train/evaluation split from dataset."""
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1 - eval_ratio))
        
        train_dataset = dataset.select(range(split_idx))
        eval_dataset = dataset.select(range(split_idx, len(dataset)))
        
        logger.info(f"Created train/eval split: {len(train_dataset)}/{len(eval_dataset)}")
        return train_dataset, eval_dataset


class DataCollator:
    """
    Custom data collator for mathematical reasoning tasks.
    Handles padding and special formatting requirements.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512, pad_to_multiple_of: int = 8):
        """Initialize data collator."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Use the standard data collator as base
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt"
        )
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features for training."""
        # Separate tensor features from metadata
        tensor_features = []
        metadata = []
        
        for feature in features:
            tensor_feature = {k: v for k, v in feature.items() 
                            if k in ["input_ids", "attention_mask", "labels"]}
            metadata_feature = {k: v for k, v in feature.items() 
                              if k not in ["input_ids", "attention_mask", "labels"]}
            
            tensor_features.append(tensor_feature)
            metadata.append(metadata_feature)
        
        # Use base collator for tensor features
        batch = self.base_collator(tensor_features)
        
        # Add metadata to batch
        for key in metadata[0].keys():
            batch[key] = [item[key] for item in metadata]
        
        return batch


def load_and_prepare_datasets(config: Dict, tokenizer: PreTrainedTokenizer = None) -> Dict[str, Dataset]:
    """
    Main function to load and prepare all datasets according to configuration.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for text processing (optional, will load if not provided)
        
    Returns:
        Dictionary containing prepared datasets
    """
    logger.info("Loading and preparing datasets")
    
    # Load tokenizer if not provided
    if tokenizer is None:
        from transformers import AutoTokenizer
        model_name = config.get("model", {}).get("name", "microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize dataset loader
    loader = MathDatasetLoader(config)
    
    # Get dataset names from config
    dataset_names = config.get("data", {}).get("datasets", ["gsm8k", "math"])
    
    prepared_datasets = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Processing {dataset_name}")
        
        try:
            # Load raw dataset
            if dataset_name == "svamp":
                # SVAMP is evaluation only
                raw_dataset = loader.load_dataset(dataset_name, "test")
                processed_dataset = loader.preprocess_svamp(raw_dataset)
                prepared_datasets[f"{dataset_name}_test"] = processed_dataset
            else:
                # Load train and test splits
                train_dataset = loader.load_dataset(dataset_name, "train")
                test_dataset = loader.load_dataset(dataset_name, "test")
                
                # Preprocess datasets
                if dataset_name == "gsm8k":
                    train_processed = loader.preprocess_gsm8k(train_dataset)
                    test_processed = loader.preprocess_gsm8k(test_dataset)
                elif dataset_name == "math":
                    train_processed = loader.preprocess_math(train_dataset)
                    test_processed = loader.preprocess_math(test_dataset)
                
                # Create train/eval split from training data
                train_split, eval_split = loader.create_train_eval_split(train_processed)
                
                # Store datasets
                prepared_datasets[f"{dataset_name}_train"] = train_split
                prepared_datasets[f"{dataset_name}_eval"] = eval_split
                prepared_datasets[f"{dataset_name}_test"] = test_processed
        
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            continue
    
    # Combine datasets if multiple training sets
    if len([k for k in prepared_datasets.keys() if k.endswith("_train")]) > 1:
        from datasets import concatenate_datasets
        
        train_datasets = [v for k, v in prepared_datasets.items() if k.endswith("_train")]
        eval_datasets = [v for k, v in prepared_datasets.items() if k.endswith("_eval")]
        
        combined_train = concatenate_datasets(train_datasets)
        combined_eval = concatenate_datasets(eval_datasets)
        
        prepared_datasets["train"] = combined_train
        prepared_datasets["eval"] = combined_eval
        
        logger.info(f"Combined datasets - Train: {len(combined_train)}, Eval: {len(combined_eval)}")
    else:
        # Single dataset case
        for k, v in list(prepared_datasets.items()):
            if k.endswith("_train"):
                prepared_datasets["train"] = v
            elif k.endswith("_eval"):
                prepared_datasets["eval"] = v
    
    logger.info(f"Dataset preparation complete. Available datasets: {list(prepared_datasets.keys())}")
    return prepared_datasets
