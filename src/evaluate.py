from unsloth import FastVisionModel
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import evaluate
import time
import json
import os
import math
import subprocess
import platform
import sys

from .dataset import get_test
from .utils import set_seed

def get_mem(): 
    return torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

def _evaluate(
    model, 
    tokenizer, 
    test_dataset, 
    batch_size: int = 8, 
    num_workers: int = 4,
    local_rank: int = -1,
    world_size: int = 1,
    ddp: bool = False,
    time_limit: int = -1,
    initial_mem: float = 0,
    post_model_mem: float = 0
):

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    FastVisionModel.for_inference(model)
    
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup distributed sampler if using DDP
    sampler = None
    if ddp:
        sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=lambda x: x,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    if local_rank in [-1, 0]:
        print(f"Starting evaluation with batch size {batch_size}...")

    gen_kwargs = {
        "max_new_tokens": 275,
        "use_cache": True,
        "temperature": 1.5,
        "min_p": 0.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    # Set device based on local_rank
    if ddp:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resume logic
    safe_model_id = model.config._name_or_path.replace("/", "_")
    checkpoint_file = f"eval_checkpoint_{safe_model_id}_rank{local_rank}.json"
    
    accumulated_predictions = []
    accumulated_references = []
    predictions_with_metadata = []  # Store predictions with index, img_id, question
    start_index = 0

    if os.path.exists(checkpoint_file):
        print(f"Rank {local_rank}: Found checkpoint {checkpoint_file}, resuming...")
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            accumulated_predictions = data.get("predictions", [])
            accumulated_references = data.get("references", [])
            predictions_with_metadata = data.get("predictions_with_metadata", [])
            start_index = len(accumulated_predictions)
        
        # Load previous results into metrics
        print(f"Rank {local_rank}: Loaded {start_index} previous results.")
        if len(accumulated_predictions) > 0:
            bleu.add_batch(predictions=accumulated_predictions, references=[[r] for r in accumulated_references])
            rouge.add_batch(predictions=accumulated_predictions, references=accumulated_references)
            meteor.add_batch(predictions=accumulated_predictions, references=accumulated_references)

    # Calculate batches to skip
    batches_to_skip = math.ceil(start_index / batch_size)
    
    start_time = time.time()
    global_idx = start_index
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Skip processed batches
        if i < batches_to_skip:
            continue
            
        # Check time limit
        if time_limit > 0 and (time.time() - start_time) > time_limit:
            print(f"Rank {local_rank}: Time limit exceeded. Saving progress and exiting...")
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "predictions": accumulated_predictions,
                    "references": accumulated_references,
                    "predictions_with_metadata": predictions_with_metadata
                }, f)

        batch_images = []
        batch_prompts = []
        batch_references = []
        batch_questions = []
        batch_img_ids = []

        for item in batch:
            user_msg = item['messages'][0]["content"]
            reference_text = item['messages'][1]["content"][0]["text"]
            question_text = user_msg[0]["text"]
            img_id = item.get("img_id", f"img_{global_idx}")

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_text}
                ]}
            ]

            text_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(text_input)
            batch_images.append(user_msg[1]["image"])
            batch_references.append(reference_text)
            batch_questions.append(question_text)
            batch_img_ids.append(img_id)
        
        inputs = tokenizer(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs
            )

        input_length = inputs.input_ids.shape[1]
        generated_text_ids = generated_ids[:, input_length:]
        batch_predictions = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)

        batch_predictions = [pred.strip() for pred in batch_predictions]  
        batch_references = [ref.strip() for ref in batch_references]

        bleu.add_batch(predictions=batch_predictions, references=[[r] for r in batch_references])
        rouge.add_batch(predictions=batch_predictions, references=batch_references)
        meteor.add_batch(predictions=batch_predictions, references=batch_references)

        # Accumulate for checkpointing
        accumulated_predictions.extend(batch_predictions)
        accumulated_references.extend(batch_references)
        
        # Store predictions with metadata
        for j, pred in enumerate(batch_predictions):
            predictions_with_metadata.append({
                "index": global_idx,
                "img_id": batch_img_ids[j],
                "question": batch_questions[j],
                "answer": pred
            })
            global_idx += 1

    # Gather predictions from all ranks if using DDP
    if ddp:
        # Synchronize all processes before gathering
        dist.barrier()
        
        # Gather predictions_with_metadata from all ranks to rank 0
        gathered_predictions = [None] * world_size
        dist.all_gather_object(gathered_predictions, predictions_with_metadata)
        
        if local_rank == 0:
            all_predictions = []
            for rank_predictions in gathered_predictions:
                all_predictions.extend(rank_predictions)
            all_predictions.sort(key=lambda x: x['index'])
            predictions_with_metadata = all_predictions
            
            print(f"Gathered {len(predictions_with_metadata)} predictions from {world_size} ranks")
    
    # Compute metrics and save JSON only on rank 0
    if local_rank in [-1, 0]:
        bleu_result = bleu.compute()
        rouge_result = rouge.compute()
        meteor_result = meteor.compute()
        
        # Calculate public scores
        bleu_score = round(bleu_result['bleu'], 4)
        rouge1_score = round(float(rouge_result['rouge1']), 4)
        rouge2_score = round(float(rouge_result['rouge2']), 4)
        rougeL_score = round(float(rouge_result['rougeL']), 4)
        meteor_score = round(float(meteor_result['meteor']), 4)
        
        public_scores = {
            'bleu': bleu_score,
            'rouge1': rouge1_score,
            'rouge2': rouge2_score,
            'rougeL': rougeL_score,
            'meteor': meteor_score
        }
        
        print("✨Public scores: ", public_scores)
        
        # Calculate timing and memory
        total_time = round(time.time() - start_time, 4)
        final_mem = round(get_mem() - post_model_mem, 2)
        model_mem_used = round(post_model_mem - initial_mem, 2)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        
        # Prepare submission info
        SUBMISSION_INFO = {
            "Participant_Names": "Dam Tien Dat",
            "Affiliations": "University of Science - VNUHCM",
            "Contact_emails": ["tdat0306@gmail.com", "23120118@student.hcmus.edu.vn"],
            "Team_Name": "Dam Tien Dat",
            "Country": "Viet Nam",
            "Notes_to_organizers": "Evaluation run"
        }
        
        # Prepare output data
        output_data = {
            "submission_info": SUBMISSION_INFO,
            "public_scores": public_scores,
            "predictions": predictions_with_metadata,
            "total_time": total_time,
            "time_per_item": total_time / len(test_dataset) if len(test_dataset) > 0 else 0,
            "memory_used_mb": final_mem,
            "model_memory_mb": model_mem_used,
            "gpu_name": gpu_name,
            "debug": {
                "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                "system": {
                    "python": platform.python_version(),
                    "os": platform.system(),
                    "platform": platform.platform(),
                    "arch": platform.machine()
                }
            }
        }
        
        # Save to JSON file
        output_file = "predictions_eval.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        
        print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
        print(f"✅ Evaluation completed successfully. Results saved to '{output_file}'.")
        
    # Cleanup checkpoint on completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def evaluate_model(
    model_id: str,
    batch_size: int = 8, 
    num_workers: int = 4,
    seed: int = 3407,
    local_rank: int = -1,
    world_size: int = 1,
    ddp: bool = False,
    time_limit: int = -1
):
    set_seed(seed)
    
    # Track memory before model loading
    initial_mem = get_mem()
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_id,
        load_in_4bit = True,
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto",
    )
    
    # Track memory after model loading
    post_model_mem = get_mem()

    test_dataset = get_test()

    _evaluate(
        model, 
        tokenizer, 
        test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        local_rank=local_rank,
        world_size=world_size,
        ddp=ddp,
        time_limit=time_limit,
        initial_mem=initial_mem,
        post_model_mem=post_model_mem
    )
