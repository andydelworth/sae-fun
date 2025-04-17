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
import h5py
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--output', type=str, default='activations_apr_10/')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb')
parser.add_argument('--token_sample_pct', type=float, default=0.1)
parser.add_argument('--layer_num', type=int, default=16)
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--output_h5', type=str, default='token_activations_500m.h5')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

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
    dataset = load_dataset(
        args.dataset,
        split='train',
        streaming=True,
        trust_remote_code=True  # FineWeb requires this
    )
    
    # Process data in batches
    world_size = dist.get_world_size()
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for streaming
        num_workers=0,  # Must be 0 for streaming
        pin_memory=True  # Can help with GPU transfer
    )

    hidden_states = defaultdict(list)
    max_seq_length = 512  # Limit sequence length to prevent OOM
    max_tokens = 500_000_000  # 500 million tokens
    token_dim = 4096  # Assuming 4096-dimensional token embeddings

    # Set random seeds for reproducibility
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    import random
    random.seed(5)

    # create HDF5 dataset
    if local_rank == 0:
        with h5py.File(args.output_h5, 'w') as h5_file:
            activations_dataset = h5_file.create_dataset(
                'activations',
                shape=(max_tokens, token_dim),
                dtype=np.float16,
                chunks=(1000, token_dim)  # Chunk size for efficient I/O
            )
            token_counter = 0
    
    # Use a simple round-robin distribution for streaming
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

        if len(hidden_states[args.layer_num]) > 4000:
            # Gather activations from all ranks
            local_acts = torch.stack(hidden_states[args.layer_num], dim=0)
            gathered_acts = [torch.zeros_like(local_acts) for _ in range(world_size)]
            dist.all_gather(gathered_acts, local_acts)
            
            # Process only on rank 0
            if local_rank == 0:
                all_acts = torch.cat(gathered_acts, dim=0)
                available_space = max_tokens - token_counter
                num_to_write = min(available_space, all_acts.shape[0])
                if num_to_write > 0:
                    with h5py.File(args.output_h5, 'a') as h5_file:
                        h5_file['activations'][token_counter:token_counter + num_to_write] = all_acts[:num_to_write].cpu().to(torch.float16).numpy()
                    token_counter += num_to_write
            
            hidden_states = defaultdict(list)
            torch.cuda.empty_cache()

        # Inside the main loop, every iteration
        if local_rank == 0:
            if token_counter >= max_tokens:
                break

    if local_rank == 0:
        print(f"Reached maximum token limit of {max_tokens}")
        print(f"Current token counter: {token_counter}")
        print("HDF5 file closed automatically by context manager")

if __name__ == "__main__":
    main()

