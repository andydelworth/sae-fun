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
parser.add_argument('--output_h5', type=str, default='token_activations.h5')
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
    max_tokens = 330_000_000  # 330 million tokens
    token_dim = 4096  # Assuming 4096-dimensional token embeddings

    # Set random seeds for reproducibility
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    import random
    random.seed(5)

    # create HDF5 dataset
    if local_rank == 0:
        h5_file = h5py.File(args.output_h5, 'w')
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
                # Flatten and save all gathered activations
                all_acts = torch.cat(gathered_acts, dim=0)
                num_new_tokens = all_acts.shape[0]
                
                if token_counter + num_new_tokens > max_tokens:
                    print(f"Reached maximum token limit of {max_tokens}")
                    print(f"Current token counter: {token_counter}")
                    print("Closing HDF5 file...")
                    h5_file.close()
                    sys.exit(1)
                
                # Convert BFloat16 to float16 before saving to HDF5
                activations_dataset[token_counter:token_counter + num_new_tokens] = all_acts.cpu().to(torch.float16).numpy()
                token_counter += num_new_tokens
                
                if token_counter % 1000000 == 0:  # Print progress every 1M tokens
                    print(f"Processed {token_counter} tokens")
            
            hidden_states = defaultdict(list)
            torch.cuda.empty_cache()

    if local_rank == 0:
        h5_file.close()

if __name__ == "__main__":
    main()

