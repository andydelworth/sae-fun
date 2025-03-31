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
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--output', type=str, default='activations/')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--dataset', type=str, default='monology/pile-uncopyrighted')
args = parser.parse_args()

layers_of_interest = [16, 24]
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
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    
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
        ) # TODO - maybe we don't want to waste the rest of the seq

        input_ids = tokens['input_ids'].to(local_rank)
        attention_mask = tokens['attention_mask'].to(local_rank)
        
        with torch.cuda.amp.autocast():  # Use mixed precision
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        batch_hidden_states = outputs.hidden_states

        for layer in layers_of_interest:
            hidden_states[layer].extend(batch_hidden_states[layer]) # note that first hidden state is just the embeddings

        # Get predicted tokens by taking argmax of logits
        predicted_tokens = torch.argmax(outputs.logits, dim=-1)
        
        # Convert tokens back to text
        generated_text = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)

        total_examples += len(batch['text'])
        if total_examples >= 100:
            for layer in layers_of_interest:
                # Include rank in batch number to prevent overwrites
                global_batch = (i // world_size) * world_size + local_rank
                torch.save(torch.cat(hidden_states[layer], dim=0), f'{args.output}/layer_{layer}_batch_{global_batch}.pt')
            hidden_states = defaultdict(list)
            torch.cuda.empty_cache()  # Clear cache after saving
        
        if total_examples >= 5000:
            break

if __name__ == "__main__":
    main()

