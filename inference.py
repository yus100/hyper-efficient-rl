"""Simple inference script for the fine-tuned Qwen model."""

import torch
from unsloth import FastLanguageModel

def load_finetuned_model(model_path="./qwen3-gsm8k-lora"):
    """Load the fine-tuned model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, question):
    """Generate response for a math question."""
    prompt = f"<|im_start|>user\nSolve this math problem step by step:\n\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    response = response.split("<|im_start|>assistant\n")[-1]
    
    return response

def main():
    """Test the fine-tuned model."""
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    
    # Test questions
    test_questions = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A store sells pencils for $0.25 each and erasers for $0.75 each. If someone buys 4 pencils and 2 erasers, how much do they spend in total?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Q: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"A: {response}")

if __name__ == "__main__":
    main()
