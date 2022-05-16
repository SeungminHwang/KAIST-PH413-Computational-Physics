import numpy as np
import torch
from torch.utils.data import Dataset

class COVID_SIR_Dataset(Dataset):
    def __init__(self, x, y, s0, aux, device):
        self.x_data = torch.as_tensor(
            np.array(x).astype(np.float32)
        ).to(device)
        
        self.y_data = torch.as_tensor(
            np.array(y).astype(np.float32)
        ).to(device)
        
        self.s0_data = torch.as_tensor(
            np.array(s0).astype(np.float32)
        ).to(device)
        
        self.aux_data = torch.as_tensor(
            np.array(aux).astype(np.int32)
        ).to(device)
        
        #print(aux)
        
        
        self.n_sample = s0.shape[0]
        
        #print(self.x_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.s0_data[index], self.aux_data[index]
    
    def __len__(self):
        return self.n_sample
    def getInputSize(self):
        pass
    def getOutputSize(self):
        pass
    
    