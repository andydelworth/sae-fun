from concurrent.futures import ThreadPoolExecutor
import shutil
import pickle
from tqdm import tqdm
import transformers
import datasets
import torch
import argparse
from sae import SAE
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from queue import Queue
from threading import Thread

save_queue = Queue(maxsize=64)  # blocks if too much backlog
executor = ThreadPoolExecutor(max_workers=16)
SAE_DEVICE = 'cuda:1'

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(local_rank, args):
    sd = torch.load(args.sae_path)['model_state_dict']
    sae = SAE(
        input_dim=sd['decoder.0.weight'].shape[0],
        hidden_dim=sd['decoder.0.weight'].shape[1]
    )
    sae.load_state_dict(sd)
    sae = sae.to(torch.float32)
    sae = sae.to(SAE_DEVICE)
    sae.eval()
    sae = DDP(sae, device_ids=[SAE_DEVICE])
    # if SAE_DEVICE == :
    print(f'SAE loaded with {sum(p.numel() for p in sae.parameters())} parameters')

    lm = transformers.AutoModelForCausalLM.from_pretrained(args.model_name).to(local_rank)
    lm.eval()
    lm = DDP(lm, device_ids=[local_rank])
    if local_rank == 0:
        print(f'LM loaded with {sum(p.numel() for p in lm.parameters())} parameters')

    return sae, lm

def get_sae_activations(text, sae, lm, tokenizer, max_seq_length, local_rank):

    tokens = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length,
        return_tensors="pt"
    ).to(local_rank)

    input_ids = tokens['input_ids'].to(local_rank)
    attention_mask = tokens['attention_mask'].to(local_rank)
    
    with torch.cuda.amp.autocast():  # Use mixed precision
        with torch.no_grad():
            outputs = lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    batch_hidden_states = outputs.hidden_states[16].to(SAE_DEVICE) # 16th layer

    # use those as input to the sae
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            sae_activations = sae(batch_hidden_states)

    return {
        'sae_activations': sae_activations,
        'input_ids': input_ids
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct') # note - prob should have used base model
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()


    local_rank = setup_ddp()
    sae, lm = load_model(local_rank, args)
    
    # load text data
    dataset = datasets.load_dataset('HuggingFaceTB/cosmopedia',
        'web_samples_v2',
        split='train',
        streaming=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for streaming
        num_workers=0,  # Must be 0 for streaming
        pin_memory=True  # Can help with GPU transfer
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    for i, row in tqdm(enumerate(dataloader)):
        # if i >= args.max_iters:
        #     break

        # let's put a disk space limit on here
        total, used, free = shutil.disk_usage("/persist/adelworth")
        if (free / 1024**3) < 50:
            # < 50 gb free; kill it
            break

        text = row['text']
        # breakpoint()
        result = get_sae_activations(text, sae, lm, tokenizer, 512, local_rank)
        token_reconstruction, sae_activations = result['sae_activations'][0], result['sae_activations'][1]
        input_ids = result['input_ids']
        sae_activations = sae_activations * (sae_activations > 0.1) # mask out small activations
        sae_activations = sae_activations.to(torch.float16)
        # ok - let's drop the last 15/16 of features. just to save sapce.
        sae_activations = sae_activations[:,:sae_activations.shape[1] // 16]

        '''
        question - is it valid to store only the last token representation of the text?

        pros:
        - much more space efficient. 
        - i wouldn't have to worry about the auto-interp methods parsing mid-document things
        cons:
        - could miss out on some feature meanings (i.e., 'surprising' tokens, incorrect punctuation token)

        let's start out with only last token - can always expand
        '''
        os.makedirs(args.output_dir, exist_ok=True)

        for batch_idx in range(sae_activations.shape[0]):
            sparsified_activation = sae_activations[batch_idx].to_sparse()
            save_queue.put((sparsified_activation, sae_activations[batch_idx], input_ids[batch_idx], text[batch_idx], args.output_dir, str(args.batch_size * i + batch_idx)))
            # we're currently saving all 512 tokens. we may or may not want to do that - is last token sufficient? let's do all fo rnow.
            # TODO - call save function
    save_queue.put(None)
    thread.join()
    executor.shutdown()

        

def save_worker():
    while True:
        item = save_queue.get()
        if item is None:
            break
        executor.submit(save_things, *item)
        save_queue.task_done()

thread = Thread(target=save_worker)
thread.start()


def save_things(sparsified, regular, ids, text, output_dir, global_idx):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{global_idx}.txt"), 'w') as f:
        f.write(text)
    torch.save(sparsified.cpu(), os.path.join(output_dir, f"{global_idx}_sparse.pt"))
    # torch.save(regular.cpu(), os.path.join(output_dir, f"{global_idx}.pt"))
    torch.save(ids.cpu(), os.path.join(output_dir, f"{global_idx}_ids.pt"))


if __name__ == '__main__':
    main()