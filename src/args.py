import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    return args