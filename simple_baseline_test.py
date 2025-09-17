#!/usr/bin/env python3
"""
Simple baseline evaluation test without heavy dependencies.
Tests the core evaluation logic.
"""

import re
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_numerical_answer(solution: str):
    """Extract numerical answer from solution text."""
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


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """Compare predicted answer with ground truth."""
    try:
        pred_num = float(str(predicted).replace(",", "").strip())
        gt_num = float(str(ground_truth).replace(",", "").strip())
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()


def mock_model_generate(problem: str) -> str:
    """Mock model generation for testing."""
    # Simple rule-based responses for testing
    problem_lower = problem.lower()
    
    if "2 + 3" in problem_lower:
        return "Let me solve this step by step. 2 + 3 = 5. The answer is 5."
    elif "15" in problem_lower and "4" in problem_lower and "×" in problem_lower:
        return "I need to multiply 15 by 4. 15 × 4 = 60. So the answer is 60."
    elif "x + 7 = 12" in problem_lower:
        return "To solve x + 7 = 12, I subtract 7 from both sides: x = 12 - 7 = 5."
    elif "25%" in problem_lower and "80" in problem_lower:
        return "25% of 80 means 0.25 × 80 = 20. The answer is 20."
    elif "√144" in problem_lower or "sqrt(144)" in problem_lower:
        return "The square root of 144 is 12, since 12 × 12 = 144."
    else:
        return "I'm not sure how to solve this problem."


def run_mock_evaluation():
    """Run a mock baseline evaluation."""
    logger.info("Running mock baseline evaluation...")
    
    # Test problems
    test_problems = [
        {"problem": "What is 2 + 3?", "answer": "5"},
        {"problem": "Calculate 15 × 4", "answer": "60"},
        {"problem": "Solve for x: x + 7 = 12", "answer": "5"},
        {"problem": "Find 25% of 80", "answer": "20"},
        {"problem": "What is √144?", "answer": "12"},
        {"problem": "What is the capital of France?", "answer": "unknown"}  # Non-math question
    ]
    
    results = []
    correct_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_problems):
        problem = test_case["problem"]
        expected = test_case["answer"]
        
        logger.info(f"\nEvaluating problem {i+1}: {problem}")
        
        # Simulate model generation with timing
        start_time = time.time()
        response = mock_model_generate(problem)
        end_time = time.time()
        
        generation_time = end_time - start_time
        total_time += generation_time
        
        # Extract predicted answer
        predicted_num = extract_numerical_answer(response)
        predicted = str(predicted_num) if predicted_num is not None else "unknown"
        
        # Check correctness
        is_correct = compare_answers(predicted, expected)
        if is_correct:
            correct_count += 1
        
        # Store results
        result = {
            "problem": problem,
            "response": response,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "time": generation_time,
            "tokens": len(response.split())  # Approximate token count
        }
        results.append(result)
        
        # Log results
        logger.info(f"  Response: {response[:100]}...")
        logger.info(f"  Expected: {expected}")
        logger.info(f"  Predicted: {predicted}")
        logger.info(f"  Correct: {'✓' if is_correct else '✗'}")
        logger.info(f"  Time: {generation_time:.3f}s")
    
    # Calculate final metrics
    accuracy = correct_count / len(test_problems)
    avg_time = total_time / len(test_problems)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    
    # Generate report
    logger.info("\n" + "="*50)
    logger.info("MOCK BASELINE EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total problems: {len(test_problems)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"Average response time: {avg_time:.3f}s")
    logger.info(f"Total evaluation time: {total_time:.3f}s")
    logger.info(f"Average response length: {avg_tokens:.1f} words")
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_problems": len(test_problems),
        "avg_time": avg_time,
        "total_time": total_time,
        "avg_tokens": avg_tokens,
        "results": results
    }


def test_evaluation_components():
    """Test individual evaluation components."""
    logger.info("Testing evaluation components...")
    
    # Test answer extraction
    test_extractions = [
        ("The answer is 42", 42.0),
        ("So we get x = 25", 25.0),
        ("Therefore, 3.14159 is the result", 3.14159),
        ("No number here", None)
    ]
    
    logger.info("\nTesting answer extraction:")
    for text, expected in test_extractions:
        result = extract_numerical_answer(text)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{text}' -> {result} (expected {expected})")
    
    # Test answer comparison
    test_comparisons = [
        ("42", "42", True),
        ("42.0", "42", True),
        ("3.14", "3.15", False),
        ("unknown", "unknown", True)
    ]
    
    logger.info("\nTesting answer comparison:")
    for pred, truth, expected in test_comparisons:
        result = compare_answers(pred, truth)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{pred}' vs '{truth}' -> {result} (expected {expected})")


def main():
    """Main test function."""
    logger.info("Starting simple baseline evaluation test...")
    
    try:
        # Test components
        test_evaluation_components()
        
        # Run mock evaluation
        results = run_mock_evaluation()
        
        logger.info("\n🎉 Simple baseline evaluation test completed!")
        
        logger.info("\n📋 What the real baseline evaluation will do:")
        logger.info("✓ Load Qwen2.5 4B model with quantization")
        logger.info("✓ Generate responses for mathematical problems")
        logger.info("✓ Extract and compare numerical answers")
        logger.info("✓ Calculate accuracy, timing, and token metrics")
        logger.info("✓ Save detailed results and generate reports")
        logger.info("✓ Support multiple datasets (GSM8K, MATH, SVAMP)")
        
        logger.info(f"\n🎯 Mock evaluation achieved {results['accuracy']:.1%} accuracy")
        logger.info("Ready to run real baseline evaluation!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
