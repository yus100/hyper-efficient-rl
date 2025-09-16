"""
Data preprocessing utilities for mathematical reasoning tasks.
Includes text cleaning, problem parsing, and solution extraction.
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sympy as sp
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class MathTextProcessor:
    """
    Text processing utilities for mathematical problems and solutions.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        # Common mathematical symbols and their LaTeX equivalents
        self.math_symbol_map = {
            "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta",
            "ε": "\\epsilon", "θ": "\\theta", "λ": "\\lambda", "μ": "\\mu",
            "π": "\\pi", "σ": "\\sigma", "τ": "\\tau", "φ": "\\phi",
            "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "±": "\\pm",
            "∞": "\\infty", "∫": "\\int", "∑": "\\sum", "∏": "\\prod",
            "√": "\\sqrt", "∂": "\\partial"
        }
        
        # Patterns for common mathematical structures
        self.fraction_pattern = re.compile(r"(\d+)/(\d+)")
        self.decimal_pattern = re.compile(r"\d+\.\d+")
        self.percentage_pattern = re.compile(r"\d+%")
        self.currency_pattern = re.compile(r"\$[\d,]+\.?\d*")
    
    def clean_problem_text(self, text: str) -> str:
        """Clean and normalize problem text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize quotation marks
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Normalize mathematical symbols
        for symbol, latex in self.math_symbol_map.items():
            text = text.replace(symbol, latex)
        
        # Standardize number formatting
        text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
        
        # Clean up spacing around mathematical operators
        text = re.sub(r'\s*([+\-*/=<>])\s*', r' \1 ', text)
        
        # Normalize fractions
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)
        
        return text.strip()
    
    def extract_numerical_answer(self, solution: str) -> Optional[float]:
        """Extract numerical answer from solution text."""
        # Try different patterns for numerical answers
        patterns = [
            r"#### ([-+]?\d*\.?\d+)",  # GSM8K format
            r"\\boxed\{([-+]?\d*\.?\d+)\}",  # MATH format
            r"(?:answer|result|final answer).*?([-+]?\d*\.?\d+)",  # General format
            r"([-+]?\d*\.?\d+)(?:\s*$|\s*\.$)",  # Number at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                try:
                    # Clean the number string
                    number_str = match.group(1).replace(",", "")
                    return float(number_str)
                except ValueError:
                    continue
        
        # Fallback: find all numbers and return the last one
        numbers = re.findall(r"[-+]?\d*\.?\d+", solution.replace(",", ""))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def parse_step_by_step_solution(self, solution: str) -> List[str]:
        """Parse solution into individual reasoning steps."""
        # Split by common step indicators
        step_indicators = [
            r"Step \d+:",
            r"\d+\.",
            r"\n\n",
            r"First,",
            r"Next,",
            r"Then,",
            r"Finally,",
            r"Therefore,",
            r"So,",
        ]
        
        # Create a combined pattern
        pattern = "|".join(f"({indicator})" for indicator in step_indicators)
        
        # Split the solution
        parts = re.split(pattern, solution, flags=re.IGNORECASE)
        
        # Clean and filter steps
        steps = []
        current_step = ""
        
        for part in parts:
            if part and part.strip():
                if any(re.match(indicator, part.strip(), re.IGNORECASE) for indicator in step_indicators):
                    if current_step.strip():
                        steps.append(current_step.strip())
                    current_step = part.strip()
                else:
                    current_step += " " + part.strip()
        
        # Add the last step
        if current_step.strip():
            steps.append(current_step.strip())
        
        # Filter out very short steps and clean
        cleaned_steps = []
        for step in steps:
            step = step.strip()
            if len(step) > 10 and not step.startswith("####"):  # Ignore GSM8K answer markers
                cleaned_steps.append(step)
        
        return cleaned_steps if cleaned_steps else [solution.strip()]
    
    def validate_mathematical_expression(self, expression: str) -> bool:
        """Validate if a string contains valid mathematical expressions."""
        try:
            # Remove common text words that might interfere
            math_expr = re.sub(r'\b(?:dollars?|cents?|years?|days?|hours?|minutes?|percent)\b', '', expression, flags=re.IGNORECASE)
            
            # Try to parse with sympy
            sp.sympify(math_expr, evaluate=False)
            return True
        except:
            # Check if it contains mathematical operators or numbers
            math_indicators = ['+', '-', '*', '/', '=', '<', '>', '(', ')', '^', 'sqrt', 'log']
            has_math = any(indicator in expression for indicator in math_indicators)
            has_numbers = bool(re.search(r'\d', expression))
            
            return has_math and has_numbers
    
    def normalize_mathematical_notation(self, text: str) -> str:
        """Normalize mathematical notation for consistency."""
        # Convert common notations to standard form
        text = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^\2', text)  # Exponents
        text = re.sub(r'sqrt\(([^)]+)\)', r'√(\1)', text)  # Square root
        text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1×\2', text)  # Multiplication
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1÷\2', text)  # Division
        
        # Normalize fractions
        text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', text)
        
        return text


class SolutionFormatter:
    """
    Formatter for mathematical solutions in different formats.
    """
    
    def __init__(self, format_type: str = "step_by_step"):
        """Initialize solution formatter."""
        self.format_type = format_type
        self.processor = MathTextProcessor()
    
    def format_for_training(self, problem: str, solution: str) -> str:
        """Format problem-solution pair for training."""
        clean_problem = self.processor.clean_problem_text(problem)
        
        if self.format_type == "step_by_step":
            steps = self.processor.parse_step_by_step_solution(solution)
            formatted_solution = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))
        else:
            formatted_solution = solution
        
        return f"Problem: {clean_problem}\n\nSolution:\n{formatted_solution}"
    
    def format_for_evaluation(self, problem: str) -> str:
        """Format problem for evaluation (without solution)."""
        clean_problem = self.processor.clean_problem_text(problem)
        return f"Solve the following mathematical problem step by step:\n\nProblem: {clean_problem}\n\nSolution:"
    
    def create_chat_format(self, problem: str, solution: str) -> Dict[str, str]:
        """Create chat-formatted training example."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Solve this math problem: {self.processor.clean_problem_text(problem)}"
                },
                {
                    "role": "assistant", 
                    "content": solution
                }
            ]
        }


class DifficultyEstimator:
    """
    Estimate difficulty of mathematical problems for curriculum learning.
    """
    
    def __init__(self, method: str = "heuristic"):
        """Initialize difficulty estimator."""
        self.method = method
        self.processor = MathTextProcessor()
        
        # Difficulty weights for different factors
        self.weights = {
            "length": 0.2,
            "math_complexity": 0.3,
            "numerical_complexity": 0.2,
            "linguistic_complexity": 0.15,
            "solution_length": 0.15
        }
    
    def estimate_heuristic_difficulty(self, problem: str, solution: str = None) -> float:
        """Estimate difficulty using heuristic methods."""
        metrics = {}
        
        # Length-based difficulty
        word_count = len(problem.split())
        metrics["length"] = min(word_count / 50, 1.0)  # Normalize to 0-1
        
        # Mathematical complexity
        metrics["math_complexity"] = self._estimate_math_complexity(problem)
        
        # Numerical complexity
        metrics["numerical_complexity"] = self._estimate_numerical_complexity(problem)
        
        # Linguistic complexity
        metrics["linguistic_complexity"] = self._estimate_linguistic_complexity(problem)
        
        # Solution length (if available)
        if solution:
            solution_words = len(solution.split())
            metrics["solution_length"] = min(solution_words / 100, 1.0)
        else:
            metrics["solution_length"] = 0.5  # Default middle value
        
        return self.combine_difficulty_metrics(metrics)
    
    def estimate_zero_shot_difficulty(self, problem: str, model, tokenizer) -> float:
        """Estimate difficulty using model's zero-shot performance."""
        try:
            import torch
            
            # Format problem for the model
            formatter = SolutionFormatter()
            prompt = formatter.format_for_evaluation(problem)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # Estimate difficulty based on response quality
            # This is a simplified heuristic - in practice, you might want more sophisticated evaluation
            response_length = len(response.split())
            has_numbers = bool(re.search(r'\d', response))
            has_math_terms = any(term in response.lower() for term in ['calculate', 'solve', 'equation', 'answer'])
            
            # Higher difficulty if model struggles (short, low-quality response)
            if response_length < 10 or not (has_numbers and has_math_terms):
                return 0.8  # High difficulty
            elif response_length > 50:
                return 0.3  # Low difficulty (model is confident)
            else:
                return 0.5  # Medium difficulty
                
        except Exception as e:
            logger.warning(f"Failed to estimate zero-shot difficulty: {e}")
            return 0.5  # Default to medium difficulty
    
    def estimate_solution_length_difficulty(self, solution: str) -> float:
        """Estimate difficulty based on solution length and complexity."""
        steps = self.processor.parse_step_by_step_solution(solution)
        num_steps = len(steps)
        
        # More steps generally indicate higher difficulty
        step_difficulty = min(num_steps / 10, 1.0)
        
        # Average step complexity
        avg_step_length = np.mean([len(step.split()) for step in steps])
        length_difficulty = min(avg_step_length / 20, 1.0)
        
        # Mathematical operations complexity
        math_ops = 0
        for step in steps:
            math_ops += len(re.findall(r'[+\-*/=<>^]', step))
        
        ops_difficulty = min(math_ops / 20, 1.0)
        
        return (step_difficulty + length_difficulty + ops_difficulty) / 3
    
    def combine_difficulty_metrics(self, metrics: Dict[str, float]) -> float:
        """Combine multiple difficulty metrics into single score."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = self.weights.get(metric, 0.1)
            total_score += value * weight
            total_weight += weight
        
        # Normalize to 0-1 range
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        return max(0.1, min(1.0, final_score))  # Ensure bounds
    
    def _estimate_math_complexity(self, problem: str) -> float:
        """Estimate mathematical complexity of the problem."""
        complexity_indicators = {
            # Basic operations
            r'[+\-]': 0.1,
            r'[*/]': 0.2,
            r'[%]': 0.2,
            
            # Advanced operations
            r'sqrt|square root': 0.3,
            r'log|logarithm': 0.4,
            r'sin|cos|tan': 0.4,
            r'integral|derivative': 0.6,
            
            # Mathematical concepts
            r'equation|solve': 0.3,
            r'polynomial|quadratic': 0.4,
            r'matrix|vector': 0.5,
            r'probability|statistics': 0.4,
            r'geometry|triangle|circle': 0.3,
            
            # Complex concepts
            r'theorem|proof': 0.6,
            r'calculus|differential': 0.7,
            r'optimization|minimize|maximize': 0.5,
        }
        
        complexity = 0.0
        problem_lower = problem.lower()
        
        for pattern, weight in complexity_indicators.items():
            if re.search(pattern, problem_lower):
                complexity += weight
        
        return min(complexity, 1.0)
    
    def _estimate_numerical_complexity(self, problem: str) -> float:
        """Estimate numerical complexity based on numbers in the problem."""
        # Count different types of numbers
        integers = len(re.findall(r'\b\d+\b', problem))
        decimals = len(re.findall(r'\d+\.\d+', problem))
        fractions = len(re.findall(r'\d+/\d+', problem))
        percentages = len(re.findall(r'\d+%', problem))
        
        # Large numbers are more complex
        large_numbers = len(re.findall(r'\b\d{4,}\b', problem))
        
        # Negative numbers add complexity
        negative_numbers = len(re.findall(r'-\d+', problem))
        
        total_numbers = integers + decimals + fractions + percentages
        
        if total_numbers == 0:
            return 0.1
        
        # Base complexity from number count
        base_complexity = min(total_numbers / 10, 0.5)
        
        # Additional complexity factors
        type_complexity = (decimals * 0.1 + fractions * 0.2 + percentages * 0.1) / total_numbers
        size_complexity = large_numbers * 0.1
        sign_complexity = negative_numbers * 0.05
        
        return min(base_complexity + type_complexity + size_complexity + sign_complexity, 1.0)
    
    def _estimate_linguistic_complexity(self, problem: str) -> float:
        """Estimate linguistic complexity of the problem statement."""
        # Sentence complexity
        sentences = problem.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        sentence_complexity = min(avg_sentence_length / 20, 0.5)
        
        # Vocabulary complexity (simple heuristic)
        complex_words = len(re.findall(r'\b\w{8,}\b', problem))
        vocab_complexity = min(complex_words / 10, 0.3)
        
        # Question complexity (multiple questions are harder)
        question_marks = problem.count('?')
        question_complexity = min(question_marks * 0.2, 0.2)
        
        return sentence_complexity + vocab_complexity + question_complexity


def preprocess_dataset_batch(batch: Dict, processor: MathTextProcessor, formatter: SolutionFormatter) -> Dict:
    """
    Preprocess a batch of dataset examples.
    
    Args:
        batch: Batch of examples from dataset
        processor: Text processor instance
        formatter: Solution formatter instance
        
    Returns:
        Preprocessed batch
    """
    processed_batch = {}
    
    # Process each field in the batch
    for key, values in batch.items():
        if key == "problem":
            processed_batch[key] = [processor.clean_problem_text(text) for text in values]
        elif key == "solution":
            processed_batch[key] = [processor.normalize_mathematical_notation(text) for text in values]
        else:
            processed_batch[key] = values
    
    return processed_batch


def create_curriculum_batches(dataset: Dataset, difficulty_estimator: DifficultyEstimator, num_levels: int = 5) -> List[Dataset]:
    """
    Create curriculum learning batches sorted by difficulty.
    
    Args:
        dataset: Dataset to organize into curriculum
        difficulty_estimator: Estimator for problem difficulty
        num_levels: Number of difficulty levels
        
    Returns:
        List of dataset batches organized by difficulty
    """
    logger.info(f"Creating curriculum with {num_levels} difficulty levels")
    
    # Add difficulty scores if not already present
    def add_difficulty_if_missing(example):
        if "difficulty" not in example or example["difficulty"] is None:
            example["difficulty"] = difficulty_estimator.estimate_heuristic_difficulty(
                example["problem"], 
                example.get("solution", "")
            )
        return example
    
    dataset_with_difficulty = dataset.map(add_difficulty_if_missing)
    
    # Sort by difficulty
    sorted_dataset = dataset_with_difficulty.sort("difficulty")
    
    # Split into curriculum levels
    total_samples = len(sorted_dataset)
    samples_per_level = total_samples // num_levels
    
    curriculum_batches = []
    
    for level in range(num_levels):
        start_idx = level * samples_per_level
        if level == num_levels - 1:  # Last level gets remaining samples
            end_idx = total_samples
        else:
            end_idx = (level + 1) * samples_per_level
        
        level_dataset = sorted_dataset.select(range(start_idx, end_idx))
        curriculum_batches.append(level_dataset)
        
        # Log difficulty range for this level
        difficulties = [example["difficulty"] for example in level_dataset]
        min_diff, max_diff = min(difficulties), max(difficulties)
        logger.info(f"Level {level}: {len(level_dataset)} samples, difficulty range: {min_diff:.3f}-{max_diff:.3f}")
    
    return curriculum_batches


def filter_dataset_by_criteria(dataset: Dataset, criteria: Dict[str, any]) -> Dataset:
    """
    Filter dataset based on various criteria.
    
    Args:
        dataset: Dataset to filter
        criteria: Dictionary of filtering criteria
        
    Returns:
        Filtered dataset
    """
    def meets_criteria(example):
        # Length criteria
        if "min_problem_length" in criteria:
            if len(example["problem"].split()) < criteria["min_problem_length"]:
                return False
        
        if "max_problem_length" in criteria:
            if len(example["problem"].split()) > criteria["max_problem_length"]:
                return False
        
        # Difficulty criteria
        if "min_difficulty" in criteria:
            if example.get("difficulty", 0.5) < criteria["min_difficulty"]:
                return False
        
        if "max_difficulty" in criteria:
            if example.get("difficulty", 0.5) > criteria["max_difficulty"]:
                return False
        
        # Dataset-specific criteria
        if "datasets" in criteria:
            if example.get("dataset") not in criteria["datasets"]:
                return False
        
        # Mathematical complexity criteria
        if "require_numbers" in criteria and criteria["require_numbers"]:
            if not re.search(r'\d', example["problem"]):
                return False
        
        return True
    
    filtered = dataset.filter(meets_criteria)
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered)} examples")
    return filtered
