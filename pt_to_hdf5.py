import os
import torch
import h5py
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

activation_folder = '/persist/adelworth/sae-fun/activations'
val_pct = 0.1
chunk_size = 4096
batch_size = chunk_size * 250

def load_file(fp):
    try:
        activations = torch.load(fp)
        label = 'val' if random.random() < val_pct else 'train'
        return [(label, act.cpu().numpy()) for act in activations]
    except Exception:
        return []  # silently skip corrupted files

# Get file list
file_paths = [os.path.join(root, f)
              for root, _, files in os.walk(activation_folder)
              for f in files if 'layer_24' in f]

train_buffer, val_buffer = [], []
train_ctr, val_ctr = 0, 0

with h5py.File("activations.h5", "w") as f:
    train_set = f.create_dataset("train", shape=(0, 4096), maxshape=(None, 4096),
                                 dtype='float32', chunks=(chunk_size, 4096))
    val_set = f.create_dataset("validation", shape=(0, 4096), maxshape=(None, 4096),
                               dtype='float32', chunks=(chunk_size, 4096))

    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(load_file, file_paths), total=len(file_paths)):
            for label, activation in result:
                if label == 'train':
                    train_buffer.append(activation)
                else:
                    val_buffer.append(activation)

            # write train
            while len(train_buffer) >= batch_size:
                batch = np.stack(train_buffer[:batch_size])
                train_set.resize((train_ctr + batch_size, 4096))
                train_set[train_ctr:train_ctr + batch_size] = batch
                train_ctr += batch_size
                train_buffer = train_buffer[batch_size:]

            # write val
            while len(val_buffer) >= batch_size:
                batch = np.stack(val_buffer[:batch_size])
                val_set.resize((val_ctr + batch_size, 4096))
                val_set[val_ctr:val_ctr + batch_size] = batch
                val_ctr += batch_size
                val_buffer = val_buffer[batch_size:]

    # write remaining
    if train_buffer:
        batch = np.stack(train_buffer)
        train_set.resize((train_ctr + batch.shape[0], 4096))
        train_set[train_ctr:train_ctr + batch.shape[0]] = batch

    if val_buffer:
        batch = np.stack(val_buffer)
        val_set.resize((val_ctr + batch.shape[0], 4096))
        val_set[val_ctr:val_ctr + batch.shape[0]] = batch