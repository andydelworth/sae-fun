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

from queue import PriorityQueue

class FixedSizePriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.q = PriorityQueue()

    def push(self, item):
        if self.q.qsize() < self.max_size:
            self.q.put(item)
        else:
            # Check if item is better than smallest
            smallest = self.q.get()
            if item > smallest:
                self.q.put(item)
            else:
                self.q.put(smallest)  # Keep the old one

    def get_sorted(self, reverse=False):
        items = []
        while not self.q.empty():
            items.append(self.q.get())
        if not reverse:
            for item in items:
                self.q.put(item)
        else:
            for item in reversed(items):
                self.q.put(item)
        return sorted(items, reverse=reverse)

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

fsh = FixedSizePriorityQueue(1000)

import queue
import textwrap
q = queue.Queue()


def pull_activations(num_workers, offset):
    for file_idx, (act_f, id_f) in tqdm(enumerate(zip(act_files, id_files)), total=40000, desc=f'thread {offset}'):
        if file_idx % num_workers != offset:
            continue
        if file_idx >= 40000:
            break
        activation = torch.load(os.path.join(root, act_f)).to_dense()
        activation_nonzero = activation.nonzero()
        # breakpoint()
        if args.feature_idx >= 0:
            activation_nonzero = activation_nonzero[activation_nonzero[:, 1] == args.feature_idx]
            # TODO - why does the BOS token fire the SAE a lot? is there something wrong with the training here?
            # For now, let's mask it out
            activation_nonzero = activation_nonzero[activation_nonzero[:, 0] != 0]
        token_ids = torch.load(os.path.join(root, id_f))
        q.put((token_ids, activation_nonzero, activation))
        # for seq_idx, feature_idx in activation_nonzero:
        #     mapping[feature_idx].append((file_idx, seq_idx)) # note - right now since the data is wrong, batch_idx = token_idx :(
        #     annotated = annotate_text(tokenizer, token_ids, seq_idx, activation[seq_idx,feature_idx], feature_idx)
        #     fsh.push((activation[seq_idx,feature_idx], annotated))


def write_annotations():
    data = q.get()
    if data is None:
        return
    else:
        token_ids, activation_nonzero, activation = q.get()
    for seq_idx, feature_idx in activation_nonzero:
        # mapping[feature_idx].append((file_idx, seq_idx)) # note - right now since the data is wrong, batch_idx = token_idx :(
        annotated = annotate_text(tokenizer, token_ids, seq_idx, activation[seq_idx,feature_idx], feature_idx)
        fsh.push((activation[seq_idx,feature_idx], annotated))



pullers = [threading.Thread(target=pull_activations, args=(16, i,)) for i in range(16)]
pushers = [threading.Thread(target=write_annotations) for _ in range(16)]

for pull_t, push_t in zip(pullers, pushers):
    pull_t.start()
    push_t.start()

for t in pullers:
    t.join()

for _ in range(16):
    q.put(None)

for t in pushers:
    t.join()

breakpoint()

# for file_idx, (act_f, id_f) in tqdm(enumerate(zip(act_files, id_files)), total=40000):
#     if file_idx >= 1000:
#         break
#     activation = torch.load(os.path.join(root, act_f)).to_dense()
#     activation_nonzero = activation.nonzero()
#     # breakpoint()
#     if args.feature_idx >= 0:
#         activation_nonzero = activation_nonzero[activation_nonzero[:, 1] == args.feature_idx]
#         # TODO - why does the BOS token fire the SAE a lot? is there something wrong with the training here?
#         # For now, let's mask it out
#         activation_nonzero = activation_nonzero[activation_nonzero[:, 0] != 0]
#     token_ids = torch.load(os.path.join(root, id_f))
#     for seq_idx, feature_idx in activation_nonzero:
#         mapping[feature_idx].append((file_idx, seq_idx)) # note - right now since the data is wrong, batch_idx = token_idx :(
#         annotated = annotate_text(tokenizer, token_ids, seq_idx, activation[seq_idx,feature_idx], feature_idx)
#         fsh.push((activation[seq_idx,feature_idx], annotated))
        # os.makedirs(f'annotations/{feature_idx}', exist_ok=True)
        # with open(f'annotations/{feature_idx}/{file_idx}_{seq_idx}.txt', 'w') as f:
            # f.write(annotated)



# write the top 1k texts to a file
os.makedirs(f'annotations/{feature_idx}', exist_ok=True)
with open(f'annotations/{feature_idx}/top_1k.txt', 'w') as f:
    for _, annotated in fsh.get_sorted(reverse=True):
        f.write('\n'.join(textwrap.fill(line, width=100) for line in annotated.splitlines()))
        f.write('\n' + 'barrier--------------------------------------------------------barrier' + '\n')