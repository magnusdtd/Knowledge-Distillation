from typing import Optional
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import numpy as np
import os
import wandb 

from .utils import set_seed
from .dataset import get_train_val
from .math_utils import extract_boxed_answer, is_equiv

def train(
    model_id: str,
    save_dir: str,
    save_repo_id: str,
    hf_token: str,
    wb_token: str,
    val_size: int,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    per_device_eval_batch_size: int = 8,
    eval_accumulation_steps: int = 4,
    max_steps: Optional[int] = None,
    num_train_epochs: int = 1,
    save_steps: int = 100, 
    eval_steps: int = 100,
    early_stopping_patience: int = 5, 
    seed: int = 3407,
    local_rank: int = -1,
    world_size: int = 1,
    ddp: bool = False,
    logging_steps: int = 1,
    dataset_num_proc: int = 4,
    warmup_steps: int = 5,
    max_seq_length: int = 2048,
):    
    set_seed(seed)

    # Configure Weights & Biases
    if wb_token:
        wandb.login(key=wb_token)
        report_to = "wandb"
    else:
        report_to = "none"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = seed,
        use_rslora = False, 
        loftq_config = None, 
    )

    # Data
    train_dataset, val_dataset = get_train_val(val_size, tokenizer)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = np.where(labels != -100, labels, pad_token_id)
        preds = np.where(labels != pad_token_id, preds, pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute Pass@1 for mathematical reasoning
        metrics = compute_pass_at_1(decoded_preds, decoded_labels)
        return {"pass@1": metrics["pass@1"]}

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps if num_train_epochs else -1,
            num_train_epochs = num_train_epochs,
            learning_rate = 2e-4,
            logging_steps = logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = seed,
            output_dir = "outputs",
            report_to = report_to,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            dataset_num_proc = dataset_num_proc,
            
            # Validation Strategy Settings
            save_strategy = "steps",
            save_steps = save_steps, 
            eval_steps = eval_steps,
            load_best_model_at_end = True,
            save_total_limit = 1, 
            fp16_full_eval = False,
            per_device_eval_batch_size = per_device_eval_batch_size,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_strategy = "steps",
            metric_for_best_model = "pass@1",
            greater_is_better = True,
            
            # DDP Configuration
            ddp_find_unused_parameters = False if ddp else None,
            local_rank = local_rank,
        ),
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = early_stopping_patience, 
        early_stopping_threshold = 1e-6,
    )
    trainer.add_callback(early_stopping_callback)

    # Show current memory stats (only on rank 0)
    if local_rank in [-1, 0]:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final memory and time stats (only on rank 0)
    if local_rank in [-1, 0]:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save model only on rank 0
    if local_rank in [-1, 0]:
        model.save_pretrained_merged(save_dir, tokenizer, save_method = "merged_16bit",)
        model.push_to_hub_merged(save_repo_id, tokenizer, save_method = "merged_16bit", private=False, token=hf_token)