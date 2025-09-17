"""Evaluate models on GSM8K dataset."""
import torch
print(f'hello {torch.cuda.is_available()}')
import re
from unsloth import FastLanguageModel
from data_loader import load_gsm8k_data
from tqdm import tqdm



def load_base_model():
    """Load the base Qwen3-1.7B model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-1.7B-Base",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def extract_answer(text):
    """Extract numerical answer from model response."""
    # Look for patterns like "The answer is X" or numbers at the end
    patterns = [
        r"the answer is\s*\$?(\d+(?:\.\d+)?)",
        r"answer:\s*\$?(\d+(?:\.\d+)?)",
        r"\$(\d+(?:\.\d+)?)(?:\s|$)",
        r"(\d+(?:\.\d+)?)\s*dollars?",
        r"(\d+(?:\.\d+)?)$"
    ]
    
    text = text.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    
    # Fallback: find the last number in the text
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])
    
    return None

def extract_ground_truth(answer_text):
    """Extract numerical answer from GSM8K ground truth."""
    # GSM8K answers end with #### followed by the number
    match = re.search(r"####\s*(\d+(?:\.\d+)?)", answer_text)
    if match:
        return float(match.group(1))
    return None

def evaluate_model(model, tokenizer, dataset, max_samples=100):
    """Evaluate model on dataset."""
    correct = 0
    total = 0
    
    print(f"Evaluating on {min(max_samples, len(dataset))} samples...")
    
    for i, example in enumerate(tqdm(dataset)):
        if i >= max_samples:
            break
            
        # Extract question from formatted text
        text = example["text"]
        if "<|im_start|>user\n" in text:
            question = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            question = question.replace("Solve this math problem step by step:\n\n", "")
            ground_truth_text = text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
        else:
            # Fallback for different format
            continue
        
        # Generate model response
        prompt = f"Solve this math problem step by step:\n\n{question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Extract answers
        predicted_answer = extract_answer(response)
        ground_truth = extract_ground_truth(ground_truth_text)
        
        if predicted_answer is not None and ground_truth is not None:
            if abs(predicted_answer - ground_truth) < 0.01:  # Allow small floating point errors
                correct += 1
            total += 1
            
            if i < 3:  # Show first few examples
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question[:100]}...")
                print(f"Model answer: {predicted_answer}")
                print(f"Ground truth: {ground_truth}")
                print(f"Correct: {abs(predicted_answer - ground_truth) < 0.01}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nResults:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy

def main():
    """Main evaluation function."""
    print("Loading base model...")
    model, tokenizer = load_base_model()
    
    print("Loading test data...")
    _, test_dataset = load_gsm8k_data()
    
    print("Evaluating base model...")
    base_accuracy = evaluate_model(model, tokenizer, test_dataset, max_samples=50)
    
    print(f"\nBase model accuracy: {base_accuracy:.2%}")

if __name__ == "__main__":
    main()
