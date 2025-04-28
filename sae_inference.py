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
def load_model(local_rank):
    model = SAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier).to(torch.float32).load_state_dict(torch.load(args.model_path))
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        print(f'Model loaded with {sum(p.numel() for p in model.parameters())} parameters')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--max_iters', type=int, default=1000)
    args = parser.parse_args()


    local_rank = setup_ddp()
    model = load_model(local_rank)
    
    # load text data
    dataset = datasets.load_dataset('m-a-p/FineFineWeb', split='train')

    for i, row in enumerate(dataset):
        if i >= args.max_iters:
            break

        text = row['text']
        print(text)
        breakpoint()




if __name__ == '__main__':
    main()