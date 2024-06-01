import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.data import Data

class Data(Data):
    def __init__(self, x, edge_index, edge_attr=None, y=None, pos=None, time=None, raw_data=None, **kwargs):
        super(Data, self).__init__(x, edge_index, y, edge_attr, pos, time, **kwargs)
        self.raw_data = raw_data
    
class Dataset(Dataset):
    def __add__(self, dataset):
        super(Dataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

