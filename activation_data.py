import os
import torch

class ActivationDataset(torch.nn.utils.Dataset):
    def __init__(self, data_root, train=True, layer_num=16):
        if train:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[1]) % 8 != layer_num]
        else:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[1]) % 8 == layer_num]

    def __getitem__(self, idx):