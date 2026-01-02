from PIL import Image
from datasets import load_dataset
from pathlib import Path
from .math_utils import extract_boxed_answer

def get_train_val(
    val_size: float,
    tokenizer,
):
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    train_dataset = dataset["train"]

    split_ds = train_dataset.train_test_split(test_size=val_size)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]

    return train_dataset, val_dataset

def get_test():
    """
    Get test dataset (placeholder for future implementation).
    """
    pass


def get_aime_2024():
    """
    Load AIME 2024 dataset and format for text-only evaluation.
    
    Returns:
        Dataset with 30 AIME 2024 problems in messages format
    """    
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    # Convert to messages format for evaluation
    formatted_data = []
    for idx, item in enumerate(dataset):
        problem = item["problem"]
        answer = item["answer"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        
        formatted_data.append({
            "messages": messages,
            "img_id": f"aime_2024_{item.get('id', idx)}"
        })
    
    return formatted_data


def get_math_500():
    """
    Load MATH-500 dataset and format for text-only evaluation.
    
    Returns:
        Dataset with 500 MATH problems in messages format
    """    
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Convert to messages format for evaluation
    formatted_data = []
    for idx, item in enumerate(dataset):
        problem = item["problem"]
        solution = item["solution"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": solution}
                ]
            }
        ]
        
        formatted_data.append({
            "messages": messages,
            "img_id": item.get("unique_id", f"math_500_{idx}")
        })
    
    return formatted_data
