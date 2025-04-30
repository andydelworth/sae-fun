import transformers
import datasets
import torch
import argparse
import os
import torch.distributed as dist

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(local_rank, args):
    sae = SAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier).to(torch.float32).load_state_dict(torch.load(args.sae_path))
    sae = sae.to(local_rank)
    sae = DDP(sae, device_ids=[local_rank])
    if local_rank == 0:
        print(f'SAE loaded with {sum(p.numel() for p in model.parameters())} parameters')

    lm = transformers.AutoModelForCausalLM.from_pretrained(args.model_name).to(local_rank)
    lm = DDP(lm, device_ids=[local_rank])
    if local_rank == 0:
        print(f'LM loaded with {sum(p.numel() for p in model.parameters())} parameters')

    return sae, lm

def get_sae_activations(text, sae, lm, tokenizer, max_seq_length, local_rank):

    tokens = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(local_rank)

    hidden_state = lm(tokens, output_hidden_states=True)

    inputs = tokenizer("text", return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Embed input tokens
        hidden_states = model.model.embed_tokens(inputs.input_ids)

        # Add positional embeddings (depends on the model architecture)
        if hasattr(model.model, 'embed_positions'):
            positions = model.model.embed_positions(inputs.input_ids)
            hidden_states += positions

        # Forward pass through transformer layers up to the 16th
        for i, layer in enumerate(model.model.layers[:16]):
            hidden_states = layer(hidden_states)[0]


    breakpoint()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct') # note - prob should have used base model
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--max_iters', type=int, default=5000)
    args = parser.parse_args()


    local_rank = setup_ddp()
    sae, lm = load_model(local_rank, args)
    
    # load text data
    dataset = datasets.load_dataset('HuggingFaceTB/cosmopedia', split='train')

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    for i, row in enumerate(dataset):
        if i >= args.max_iters:
            break

        text = row['text']

        sae_activations = get_sae_activations(text, sae, lm, tokenizer, 512, local_rank)

        print(text)
        breakpoint()




if __name__ == '__main__':
    main()