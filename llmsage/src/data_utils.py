import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from get_embs import Tokenizer
from enum import Enum

class NodeType(Enum):
    Activity = 0
    Condition = 1
    Sign_Successor = 2
    Sign_Selection = 3
    Sign_Parallel = 4
    Sign_Loop = 5

class EdgeType(Enum):
    pass

from torch_geometric.data import Data
class Data(Data):
    def __init__(self, x, edge_index, y=None, edge_attr=None, pos=None, time=None, raw_data=None, **kwargs):
        super(Data, self).__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)
        self.raw_data = raw_data
    
class Dataset(Dataset):
    def __add__(self, dataset):
        super(Dataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

class RawData():
    def __init__(self, raw_data:dict) -> None:
        self.filename:str = raw_data['filename']
        self.token:list[str] = raw_data['token']
        self.bio_label:list[str] = raw_data['bio_label']
        self.text:str = raw_data['text']
        self.relation:str = raw_data['relation']
        self.data_2_mask:list[str] = raw_data['data_2_mask']
        self.signal_token_list:list[str] = raw_data['signal_token_list']
        self.data_2_mask_single_signal:list[str] = raw_data['data_2_mask_single_signal']
        self.signal_token_llm_list:list[str] = raw_data['signal_token_llm_list']
        self.data_2_mask_single_signal_llm:list[str] = raw_data['data_2_mask_single_signal_llm']

    def __str__(self) -> str:
        return f'''\nfilename: {self.filename}\n
token: {self.token}\n
bio_label: {self.bio_label}\n
text: {self.text}\n
relation: {self.relation}\n
data_2_mask: {self.data_2_mask}\n
signal_tokrn_list: {self.signal_token_list}\n
data_2_mask_single_signal: {self.data_2_mask_single_signal}\n
signal_token_llm_list: {self.signal_token_llm_list}\n
data_2_mask_single_signal_llm: {self.data_2_mask_single_signal_llm}\n'''

class DataCenter():
    def __init__(self, datasets_json:str, vocab_dir:str, vocab_len:int, embedding_size:int) -> None:
        '''
            datasets_json: str, 数据集json文件路径
            vocab_dir: str, 词表路径
            vocab_len: int, 词表长度
            embedding_size: int, 词向量维度
        '''
        self.__datasets_json:str = datasets_json
        self.__vocab_dir:str = vocab_dir
        self.__vocab_len:int = vocab_len
        self.__embedding_size:int = embedding_size
        self.__datasets_list:list[dict] = None
        self.__datasets_RawData_list:list[RawData] = None
        self.__tokenizer:Tokenizer = Tokenizer(self.__vocab_dir, self.__vocab_len, self.__embedding_size)
        self.__init_params()

    def __init_params(self):
        '''
            初始化参数
            读取数据集json文件，并将数据集json文件中每一个数据转换为RawData对象
        '''
        with open(self.__datasets_json, 'r') as f:
            self.__datasets_list = json.load(f)
            self.__datasets_RawData_list = [RawData(raw_data) for raw_data in self.__datasets_list]
            
    def __generate_dataset(self) -> Dataset:
        '''
            生成数据集
        '''
        dataset = []
        for raw_data in self.__datasets_RawData_list:
            x = self.__tokenizer.token2embedding(raw_data.data_2_mask_single_signal_llm)
            edge_index = self.__generate_edge_index(raw_data.data_2_mask_single_signal_llm)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            y = self.__generate_y(raw_data.data_2_mask_single_signal_llm)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y, raw_data=raw_data)
            dataset.append(data)
        return dataset
    def __generate_edge_index(self, data_2_mask_single_signal_llm:list[str]) -> list[list[int, int]]:
        '''
            生成邻接矩阵
            data_2_mask_single_signal_llm: list[str], 数据集数据
        '''
        cur_tokens = data_2_mask_single_signal_llm
        cur_no_activity = []
        last_activity = -1
        cur_adj_lists = []
        for j in range(0, len(cur_tokens)):
            if cur_tokens[j] != '[activity]':
                cur_no_activity.append(j)
                if last_activity != -1:
                    cur_adj_lists.append([last_activity, j])
                    cur_adj_lists.append([j, last_activity])
                    # cur_adj_lists[last_activity].add(j)
                    # cur_adj_lists[j].add(last_activity)
            elif cur_tokens[j] == '[activity]' and len(cur_no_activity) != 0:
                for k in cur_no_activity:
                    cur_adj_lists.append([k, j])
                    cur_adj_lists.append([j, k])
                    # cur_adj_lists[k].add(j)
                    # cur_adj_lists[j].add(k)
                last_activity = j
                cur_no_activity = []
            elif cur_tokens[j] == '[activity]' and len(cur_no_activity) == 0:
                #两个activity直接相连的时候，或许可以考虑在中间加个什么标识
                if last_activity != -1:
                    cur_adj_lists.append([last_activity, j])
                    cur_adj_lists.append([j, last_activity])
                    # cur_adj_lists[last_activity].add(j)
                    # cur_adj_lists[j].add(last_activity)
                last_activity = j
        return cur_adj_lists
    
    def __generate_y(self, data_2_mask_single_signal_llm:list[str]) -> list[int]:
        def get_y_category(token:str) -> int:
            if token == '[activity]':
                return NodeType.Activity.value
            elif token == '[condition]':
                return NodeType.Condition.value
            elif token == '[sign-successor]':
                return NodeType.Sign_Successor.value
            elif token == '[sign-selection]':
                return NodeType.Sign_Selection.value
            elif token == '[sign-parallel]':
                return NodeType.Sign_Parallel.value
            elif token == '[sign-loop]':
                return NodeType.Sign_Loop.value
            else:
                raise Exception(f'{token} is not in [activity, condition, sign-successor, sign-selection, sign-parallel, sign-loop]')
        return [get_y_category(token) for token in data_2_mask_single_signal_llm]
    def get_dataloader(self, batch_size:int=1, shuffle:bool=True) -> DataLoader:
        '''
            返回DataLoader
            batch_size: 批大小
            shuffle: 是否打乱
        '''
        dataset = self.__generate_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

import Parser
if __name__ == '__main__':
    args = Parser.args
    data_center = DataCenter(args.datasets_json, args.vocab_dir, args.vocab_len, args.embedding_size)
    dataloader = data_center.get_dataloader(args.batch_size, args.shuffle)
    for batch in dataloader:
        print(batch)
        print(batch.raw_data[0].filename)
        print(batch.raw_data[0])
        break
    
