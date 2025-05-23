import torch
import os
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import heapq
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--feature_idx', type=int, default=-1)
args = parser.parse_args()

root = '/persist/adelworth/sae-fun/inference_output_5_11_working'

files = os.listdir('/persist/adelworth/sae-fun/inference_output_5_11_working')

txt_files = sorted([f for f in files if '.txt' in f])
id_files = sorted([f for f in files if '_ids.pt' in f])
act_files = sorted([f for f in files if '_sparse.pt' in f])

class FixedSizeHeap:
    def __init__(self, max_size):
        self.heap = []
        self.max_size = max_size

    def push(self, item):
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def get_sorted(self, reverse=False):
        return sorted(self.heap, reverse=reverse)

def highlight_token_by_index(tokenizer, ids, highlight_idx, highlight_fn):
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    tokens[highlight_idx] = highlight_fn(tokens[highlight_idx]).replace('Ġ', '')
    return tokenizer.convert_tokens_to_string(tokens)

def annotate_text(tokenizer, token_ids, highlight_idx, activation_magnitude, feature_idx):
    highlight_fn = lambda x: f'[FEATURE {feature_idx}: {activation_magnitude}]{x}[END FEATURE]'
    annotated_text = highlight_token_by_index(tokenizer, token_ids, highlight_idx, highlight_fn)
    return annotated_text


# now that we have loaded the data, let's process into activations

assert len(txt_files) == len(id_files) == len(act_files)

mapping = defaultdict(list)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

print(len(txt_files))

os.makedirs('annotations', exist_ok=True)

fsh = FixedSizeHeap(1000)
fsh_m = defaultdict(lambda: FixedSizeHeap(1000))

import queue
import textwrap
queue = queue.Queue()


for file_idx, (act_f, id_f) in tqdm(enumerate(zip(act_files, id_files)), total=40000):
    if file_idx >= 40000:
        break
    activation = torch.load(os.path.join(root, act_f)).to_dense()
    activation_nonzero = activation.nonzero()
    # breakpoint()
    if args.feature_idx >= 0:
        feature_idx_set = torch.arange(start=40, end=49)
        activation_nonzero = activation_nonzero[
            (activation_nonzero[:, 1:2] == feature_idx_set).any(dim=1)
        ]
        # TODO - why does the BOS token fire the SAE a lot? is there something wrong with the training here?
        # For now, let's mask it out
        activation_nonzero = activation_nonzero[activation_nonzero[:, 0] != 0]
    token_ids = torch.load(os.path.join(root, id_f))
    for seq_idx, feature_idx in activation_nonzero:
        mapping[feature_idx].append((file_idx, seq_idx)) # note - right now since the data is wrong, batch_idx = token_idx :(
        annotated = annotate_text(tokenizer, token_ids, seq_idx, activation[seq_idx,feature_idx], feature_idx)
        fsh_m[feature_idx.item()].push((activation[seq_idx,feature_idx], annotated))
        # os.makedirs(f'annotations/{feature_idx}', exist_ok=True)
        # with open(f'annotations/{feature_idx}/{file_idx}_{seq_idx}.txt', 'w') as f:
            # f.write(annotated)

# write the top 1k texts to a file
for f_idx in range(40, 49):
    os.makedirs(f'annotations/{f_idx}', exist_ok=True)
    with open(f'annotations/{f_idx}/top_1k.txt', 'w') as f:
        for _, annotated in fsh_m[f_idx].get_sorted(reverse=True):
            f.write('\n'.join(textwrap.fill(line, width=100) for line in annotated.splitlines()))
            f.write('\n' + 'barrier--------------------------------------------------------barrier' + '\n')