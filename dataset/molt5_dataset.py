import torch
import re
from torch.utils.data import Dataset    

class Mol2CaptionDataset(Dataset):
    def __init__(self, raw_folder, pro_folder, mode):
        raw_file = raw_folder + '/{}.txt'.format(mode)
        with open(raw_file, 'r') as f:
            lines = f.readlines()

        lines = lines[1:]
        self.data = []
        for line in lines:
            temp = line.strip().split('\t')
            self.data.append([temp[-2], temp[-1]])

        pro_file = pro_folder + '{}.txt'.format(mode)
        with open(pro_file, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        
        for idx in range(len(lines)):
            temp = lines[idx].strip().split('\t')
            try:
                self.data[idx].extend([temp[-2], temp[-1]])
            except:
                print(idx)
                exit(0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # format: [molecule, caption, pred_caption, pred_molecule]
        return self.data[idx]