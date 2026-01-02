from PIL import Image
from datasets import load_dataset
from pathlib import Path


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

def get_test(

):
    pass

# AIME 2024 pass@1
def get_aime_2024():
    pass

# AIME 2024 cons@64

# MATH-500 pass@1

# GPQA Diamond pass@1

# LiveCodeBench pass@1

# CodeForces rating
