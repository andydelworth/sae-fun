from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import h5py

class ActivationDataset(Dataset):
    def __init__(self, data_root, layer_num, train=True):
        if train:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 != 0]
        else:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 == 0]

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
        return batch[inner_batch_idx]
    
    def __len__(self):
        if len(self.idx_map) == 0:
            return 0
        return max(self.idx_map.keys()) + 1

class HDF5ActivationDataset(Dataset):
    def __init__(self, hdf5_file, split="train"):
        """
        PyTorch Dataset for HDF5 storage.
        
        Args:
            hdf5_file (str): Path to the HDF5 file.
            split (str): Dataset split, "train" or "test".
        """
        self.hdf5_file = hdf5_file
        self.split = split

        # Open file to get dataset size
        with h5py.File(self.hdf5_file, "r") as f:
            self.length = len(f[self.split])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.hdf5_file, "r") as f:
            tensor = torch.tensor(f[self.split][index])  # Load tensor lazily
        return tensor

def get_data_loaders(activation_dir, layer_num, batch_size=40, num_workers=4, dist=False):
    # train_set = ActivationDataset(activation_dir, layer_num, train=True)
    train_set = ActivationDataset(activation_dir, layer_num, train=False)
    # train_set = HDF5ActivationDataset(activation_dir, split='train')
    # validation_set = HDF5ActivationDataset(activation_dir, split='validation')

    if dist:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        # validation_sampler = DistributedSampler(validation_set, shuffle=False)
    else:
        train_sampler = None
        # validation_sampler = None

    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, sampler=train_sampler, prefetch_factor=4, pin_memory=True)
    # validation_loader = DataLoader(validation_set, batch_size, num_workers=num_workers, sampler=validation_sampler, prefetch_factor=4, pin_memory=True)

    return train_loader, None

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    import argparse
    import torch.distributed as dist
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_dir', type=str, default='april_activations_v3')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--layer_num', type=int, default=16)
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    train_loader, validation_loader = get_data_loaders(args.activation_dir, args.layer_num, args.batch_size, args.num_workers, dist=True)

    if local_rank == 0:
        print("\nTiming train loader...")
    
    start_time = time.time()
    for i, batch in tqdm(enumerate(train_loader), disable=local_rank != 0):
        batch = batch.to(local_rank)
        if i > 1000:
            break
    train_time = time.time() - start_time

    # Gather timing stats from all processes
    train_times = [None] * dist.get_world_size()
    dist.all_gather_object(train_times, train_time)

    if local_rank == 0:
        avg_time = sum(train_times) / len(train_times)
        print(f"\nTrain loader sample took {avg_time:.2f}s averaged across {len(train_times)} processes")
        print(f"Effective throughput: {1000 * args.batch_size * len(train_times) / avg_time:.2f} samples/sec")