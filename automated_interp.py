import torch
import os
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_idx', type=int, default=-1)
args = parser.parse_args()

root = '/persist/adelworth/sae-fun/inference_output_5_11_working'

files = os.listdir('/persist/adelworth/sae-fun/inference_output_5_11_working')

txt_files = sorted([f for f in files if '.txt' in f])
id_files = sorted([f for f in files if '_ids.pt' in f])
act_files = sorted([f for f in files if '_sparse.pt' in f])

def highlight_token_by_index(tokenizer, ids, highlight_idx, highlight_fn):
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    tokens[highlight_idx] = highlight_fn(tokens[highlight_idx])
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

for file_idx, (act_f, id_f) in tqdm(enumerate(zip(act_files, id_files))):
    activation = torch.load(os.path.join(root, act_f)).to_dense()
    activation_nonzero = activation.nonzero()
    if args.feature_idx >= 0:
        activation_nonzero = activation_nonzero[activation_nonzero[:, 1] == args.feature_idx]
    token_ids = torch.load(os.path.join(root, id_f))
    for seq_idx, feature_idx in activation_nonzero:
        mapping[feature_idx].append((file_idx, seq_idx)) # note - right now since the data is wrong, batch_idx = token_idx :(
        annotated = annotate_text(tokenizer, token_ids, seq_idx, activation[seq_idx,feature_idx], feature_idx)
        os.makedirs(f'annotations/{feature_idx}', exist_ok=True)
        with open(f'annotations/{feature_idx}/{file_idx}_{seq_idx}.txt', 'w') as f:
            f.write(annotated)


