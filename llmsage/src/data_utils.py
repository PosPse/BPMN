import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import json

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
    

class DataCenter():
    def __init__(self, datasets_json:str, vocab_dir:str, vocab_len:int, embedding_size:int) -> None:
        self.__datasets_json = datasets_json
        self.vocab_dir = vocab_dir
        self.vocab_len = vocab_len
        self.embedding_size = embedding_size
        self.datasets_list = None
        self.__init_params()

    def __init_params(self):
        with open(self.__datasets_json, 'r') as f:
            self.datasets_list = json.load(f)
        # self.tokenizer = Tokenizer(self.vocab_dir, self.vocab_len)
        # self.tokenizer.load_vocab()
        # self.tokenizer.load_embedding()
        # self.tokenizer.load_embedding_matrix()
        # self.tokenizer.load_embedding_matrix_2()

    def get_dataloader(self):
        pass

    
