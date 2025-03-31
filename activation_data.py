from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ActivationDataset(Dataset):
    def __init__(self, data_root, layer_num, train=True):
        if train:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 != 0]
        else:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 == 0]

        self.files = self.files # debugging

        self.idx_map = {}
        total_len = 0
        for fp in tqdm(self.files, desc='Preprocessing activations'):
        # for fp in self.files:
            try:
                activations = torch.load(fp)
            except Exception as e:
                print(f'Error loading {fp}: {e}')
                raise e
            for j in range(len(activations)):
                self.idx_map[j + total_len] = (fp, j)
            total_len += len(activations)
        
    def __getitem__(self, idx):
        fp, inner_batch_idx = self.idx_map[idx]
        batch = torch.load(fp, map_location='cpu')
        # Keep data on CPU, DataLoader will handle GPU transfer
        return batch[inner_batch_idx]
    
    def __len__(self):
        if len(self.idx_map) == 0:
            return 0
        return max(self.idx_map.keys()) + 1


def get_data_loaders(activation_dir, layer_num, batch_size=8, num_workers=4):
    train_set = ActivationDataset(activation_dir, layer_num, train=True)
    validation_set = ActivationDataset(activation_dir, layer_num, train=False)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    validation_sampler = DistributedSampler(validation_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, sampler=train_sampler, prefetch_factor=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size, num_workers=num_workers, sampler=validation_sampler, prefetch_factor=4, pin_memory=True)
    return train_loader, validation_loader