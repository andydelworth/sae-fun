import os
import torch
import h5py
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

activation_folder = '/persist/adelworth/sae-fun/april_activations_v3'
val_pct = 0.1
chunk_size = 4096
batch_size = chunk_size * 250

def count_activations(fp):
    try:
        return len(torch.load(fp))
    except Exception:
        return 0

def load_file(fp):
    try:
        activations = torch.load(fp)
        label = 'val' if random.random() < val_pct else 'train'
        return [(label, act.cpu().numpy()) for act in activations]
    except Exception:
        return []

# Get relevant file paths
file_paths = [os.path.join(root, f)
              for root, _, files in os.walk(activation_folder)
              for f in files if 'layer_16' in f]

# Count total activations
print('counting activations!')
with Pool(cpu_count()) as pool:
    counts = [count_activations(fp) for fp in tqdm(file_paths, total=len(file_paths))]

print('counts calculated!')
total_activations = sum(counts)
print(total_activations)
n_val = int(total_activations * val_pct)
n_train = total_activations - n_val
n_val = int(n_val * 1.001)
n_train = int(n_train * 1.001) # hacky way to protect against random sample lol

# Allocate HDF5 datasets
print('allocating dset!')
with h5py.File("activations.h5", "w") as f:
    train_set = f.create_dataset("train", shape=(n_train, 4096), dtype='float32',
                                 chunks=(chunk_size, 4096))
    val_set = f.create_dataset("validation", shape=(n_val, 4096), dtype='float32',
                               chunks=(chunk_size, 4096))

    train_ctr, val_ctr = 0, 0

    print('creating pool!')
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(load_file, file_paths), total=len(file_paths)):
            for label, activation in result:
                if label == 'train':
                    train_set[train_ctr] = activation
                    train_ctr += 1
                else:
                    val_set[val_ctr] = activation
                    val_ctr += 1

    # Resize down if we overestimated due to randomness
    if train_ctr < n_train:
        train_set.resize((train_ctr, 4096))
    if val_ctr < n_val:
        val_set.resize((val_ctr, 4096))
