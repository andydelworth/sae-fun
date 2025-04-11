import subprocess
import sys
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset
import os
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--output', type=str, default='activations_apr_10/')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb')
parser.add_argument('--token_sample_pct', type=float, default=0.1)
parser.add_argument('--layer_num', type=int, default=16)
parser.add_argument('--save_every', type=int, default=10)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(model_name, local_rank):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    return model

# Main training loop
def main():
    # Setup distributed training
    local_rank = setup_ddp()
    
    # Load model and tokenizer
    model = load_model(args.model, local_rank)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print('Loading ', args.dataset)
    dataset = load_dataset(args.dataset, split='train', streaming=True)
    
    # Process data in batches
    world_size = dist.get_world_size()
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False  # No shuffling for streaming
    )

    hidden_states = defaultdict(list)
    total_examples = 0
    max_seq_length = 512  # Limit sequence length to prevent OOM

    # Set random seeds for reproducibility
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    import random
    random.seed(5)

    for i, batch in tqdm(enumerate(dataloader)):
        # Skip examples based on rank to distribute data
        if i % world_size != local_rank:
            continue
            
        tokens = tokenizer(
            batch['text'], 
            padding=True, 
            truncation=True, 
            max_length=max_seq_length,
            return_tensors="pt"
        )

        input_ids = tokens['input_ids'].to(local_rank)
        attention_mask = tokens['attention_mask'].to(local_rank)
        
        with torch.cuda.amp.autocast():  # Use mixed precision
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        batch_hidden_states = outputs.hidden_states

        max_range = batch_hidden_states[args.layer_num].shape[0] * batch_hidden_states[args.layer_num].shape[1]
        idcs = torch.randperm(max_range)[:int(max_range*args.token_sample_pct)]
        # map the integer idcs back to 2d
        batch_idx = idcs // batch_hidden_states[args.layer_num].shape[1]
        token_idx = idcs % batch_hidden_states[args.layer_num].shape[1]
        random_activation_set = batch_hidden_states[args.layer_num][batch_idx,token_idx,:].bfloat16()
        hidden_states[args.layer_num].extend(random_activation_set) # note that first hidden state is just the embeddings

        if (i + 1) % args.save_every == 0:
            global_batch = (i // args.save_every) * world_size + local_rank
            torch.save(torch.stack(hidden_states[args.layer_num], dim=0), f'{args.output}/layer_{args.layer_num}_batch_{global_batch}.pt')
            hidden_states = defaultdict(list)
        
            result = subprocess.run(['du', '-sB1', args.output], capture_output=True, text=True)
            size_bytes = int(result.stdout.split()[0])
            if size_bytes >= 2.5 * 1024**4:
                print(f"Directory too large: {size_bytes / 1024**4:.2f} TiB")
                sys.exit(1)

if __name__ == "__main__":
    main()

