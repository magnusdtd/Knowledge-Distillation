import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="[Knowledge Distillation] Qwen2.5-Math-7B Train/Evaluation/Inference Script")

    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode of the script")
    
    # Model Configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",     
                        help="Model ID for training")
    parser.add_argument("--eval_model_id", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Model ID for evaluation")
    
    # Save Configuration
    parser.add_argument("--save_dir", type=str, default="Qwen2.5-Math-7B-Instruct-KD",
                        help="Directory to save the model")
    parser.add_argument("--save_repo_id", type=str, default="magnusdtd/Qwen2.5-Math-7B-Instruct-KD",
                        help="HuggingFace repository ID to push the model")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace token for pushing models")
    parser.add_argument("--wb_token", type=str, default="", help="Weights & Biases token for logging training metrics")
    
    # Training Hyperparameters
    parser.add_argument("--val_size", type=float, default=0.01, help="Validation split size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per device")
    parser.add_argument("--eval_accumulation_steps", type=int, default=4, help="Evaluation accumulation steps")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--dataset_num_proc", type=int, default=4, help="Number of processes for dataset")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    
    # Evaluation Configuration
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation mode")
    parser.add_argument("--eval_num_workers", type=int, default=4, help="Number of workers for evaluation dataloader")
    parser.add_argument("--time_limit", type=int, default=-1, help="Time limit in seconds. -1 for no limit")
    
    # DDP Configuration
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP mode")
    
    args = parser.parse_args()
    
    # Auto-detect DDP from environment variables
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.ddp = True
    
    return args