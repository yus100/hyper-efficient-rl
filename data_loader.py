"""GSM8K data loading and preprocessing for fine-tuning."""

from datasets import load_dataset
import os

def load_gsm8k_data():
    """Load GSM8K dataset and format for instruction tuning."""
    
    # Load from existing processed data if available
    train_path = "data/processed/gsm8k_train"
    test_path = "data/processed/gsm8k_test"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading existing processed GSM8K data...")
        train_dataset = load_dataset("arrow", data_files=f"{train_path}/data-00000-of-00001.arrow", split="train")
        test_dataset = load_dataset("arrow", data_files=f"{test_path}/data-00000-of-00001.arrow", split="train")
    else:
        print("Loading GSM8K from HuggingFace...")
        dataset = load_dataset("gsm8k", "main")
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
