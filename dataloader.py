import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os

def my_collate(batch):
    #data = torch.stack([torch.from_numpy(b) for b in batch], 0)
    target = [torch.from_numpy(item[:,:65]) for item in batch]
    
    #data = torch.stack(data,0)
    # print data.shape
    return torch.cat(target,0)
    



class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = os.listdir(data_dir)
        self.data_dir = data_dir

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        sample = np.load(self.data_dir + self.data_files[idx])
        
        return sample



