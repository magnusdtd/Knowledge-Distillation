import os
import torch
import torch.distributed as dist
from src.evaluate import evaluate_model
from src.args import parse_args
from src.train import train


def setup_ddp(args):
    """Initialize DDP if enabled"""
    if args.ddp:
        # Initialize the process group
        dist.init_process_group(backend="nccl")
        
        # Set the device for this process
        torch.cuda.set_device(args.local_rank)
        
        if args.local_rank == 0:
            print(f"DDP initialized with world_size={args.world_size}")
    else:
        if args.local_rank == 0 or args.local_rank == -1:
            print("Running in single-GPU mode")


def cleanup_ddp(args):
    """Cleanup DDP if enabled"""
    if args.ddp:
        dist.destroy_process_group()


def main(args):
    # Setup DDP
    setup_ddp(args)

    if args.mode == "train":
        train(
            model_id=args.model_id,
            save_dir=args.save_dir,
            save_repo_id=args.save_repo_id,
            hf_token=args.hf_token,
            wb_token=args.wb_token,
            val_size=args.val_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            eval_accumulation_steps=args.eval_accumulation_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps, 
            eval_steps=args.eval_steps,
            early_stopping_patience=args.early_stopping_patience, 
            seed=args.seed,
            local_rank=args.local_rank,
            world_size=args.world_size,
            ddp=args.ddp,
            logging_steps=args.logging_steps,
            dataset_num_proc=args.dataset_num_proc,
            warmup_steps=args.warmup_steps,
            max_seq_length=args.max_seq_length,
        )
    elif args.mode == "eval":
        evaluate_model(
            model_id=args.eval_model_id,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            seed=args.seed,
            local_rank=args.local_rank,
            world_size=args.world_size,
            ddp=args.ddp,
            time_limit=args.time_limit,
            eval_dataset=args.eval_dataset,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError("Invalid script mode")
    
    # Cleanup DDP
    cleanup_ddp(args)


if __name__ == "__main__":
    main(parse_args())
