"""GSM8K data loading and preprocessing for fine-tuning."""

from datasets import load_dataset
import os

def load_gsm8k_data():
    """Load GSM8K dataset and format for instruction tuning."""
    
    # Always load from HuggingFace to avoid compatibility issues with processed data
    print("Loading GSM8K from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    def format_prompt(example):
        """Format GSM8K examples for instruction tuning."""
        prompt = f"Solve this math problem step by step:\n\n{example['question']}"
        response = example['answer']
        
        # Create instruction-following format
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        return {"text": text}
    
    # Format datasets
    train_formatted = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    test_formatted = test_dataset.map(format_prompt, remove_columns=test_dataset.column_names)
    
    print(f"Train samples: {len(train_formatted)}")
    print(f"Test samples: {len(test_formatted)}")
    
    return train_formatted, test_formatted

if __name__ == "__main__":
    train_data, test_data = load_gsm8k_data()
    print("Sample formatted text:")
    print(train_data[0]["text"][:500] + "...")
