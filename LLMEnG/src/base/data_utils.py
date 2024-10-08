import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from get_embs import Tokenizer
from enum import Enum
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class NodeType(Enum):
    Activity = 0
    Condition = 1
    Sign_Successor = 2
    Sign_Selection = 3
    Sign_Parallel = 4
    Sign_Loop = 5

class EdgeType(Enum):
    pass

# class Data(Data):
#     def __init__(self, x, edge_index, edge_attr=None, y=None, pos=None, time=None, raw_data=None, edge_y=None, **kwargs):
#         super(Data, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, time=time, **kwargs)
#         self.raw_data = raw_data
#         self.edge_y = edge_y
    
class Dataset(Dataset):
    def __init__(self, dataset):
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
        self.edge_y = None

#     def __str__(self) -> str:
#         return f'''\nfilename: {self.filename}\n
# token: {self.token}\n
# bio_label: {self.bio_label}\n
# text: {self.text}\n
# relation: {self.relation}\n
# data_2_mask: {self.data_2_mask}\n
# signal_tokrn_list: {self.signal_token_list}\n
# data_2_mask_single_signal: {self.data_2_mask_single_signal}\n
# signal_token_llm_list: {self.signal_token_llm_list}\n
# data_2_mask_single_signal_llm: {self.data_2_mask_single_signal_llm}\n'''

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
        self.__datasets = None
        self.__init_params()

    def __init_params(self):
        '''
            初始化参数
            读取数据集json文件，并将数据集json文件中每一个数据转换为RawData对象
        '''
        with open(self.__datasets_json, 'r') as f:
            self.__datasets_list = json.load(f)
            self.__datasets_RawData_list = [RawData(raw_data) for raw_data in self.__datasets_list]
            self.__datasets = self.__generate_dataset()
            
    def __generate_dataset(self) -> Dataset:
        '''
            生成数据集
        '''
        dataset = []
        for raw_data in self.__datasets_RawData_list:
            x = self.__tokenizer.token2embedding(raw_data.data_2_mask_single_signal_llm)
            edge_index = self.__generate_edge_index(raw_data.data_2_mask_single_signal_llm)
            y = self.__generate_y(raw_data.data_2_mask_single_signal_llm)
            num_nodes = len(raw_data.data_2_mask_single_signal_llm)
            edge_y = self.__generate_edge_y(edge_index=edge_index, num_nodes=num_nodes, y=y)
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            y = torch.tensor(y, dtype=torch.long)
            edge_y = torch.tensor(edge_y, dtype=torch.long)
            # print(edge_y)
            edge_y = SparseTensor.from_dense(edge_y)
            raw_data.edge_y = edge_y
            data = Data(x=x, edge_index=edge_index, y=y, raw_data=raw_data)
            # data = Data(x=x, edge_index=edge_index, y=y, raw_data=raw_data, edge_y=edge_y)
            dataset.append(data)
        return Dataset(dataset)
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
        cur_adj_lists = sorted(cur_adj_lists, key=lambda x: x[1])
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
        y = [get_y_category(token) for token in data_2_mask_single_signal_llm]
        return y
    
    def __generate_edge_y(self, edge_index:list[list[int]], num_nodes:int, y:list[int]) -> list[list[int]]:
        '''
            生成边标签
            edge_index: 边索引
            y: 节点标签
        '''
        def get_edge_y_category(node_i_type:int, node_j_type:int) -> int:
            edge_map = [[0,0], [0,1], [0,2],[0,3],[0,4],[0,5],[1,1],[1,2],[1,3],[1,4],[1,5],[2,2],[2,3],[2,4],[2,5],[3,3],[3,4],[3,5],[4,4],[4,5],[5,5]]
            edge = [node_i_type, node_j_type]
            edge.sort()
            return edge_map.index(edge) + 1
        
        edge_y = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for node_i, node_j in edge_index:
            node_i_type = y[node_i]
            node_j_type = y[node_j]
            edge_i_j_type = get_edge_y_category(node_i_type, node_j_type)
            edge_y[node_i][node_j] = edge_i_j_type
            edge_y[node_j][node_i] = edge_i_j_type
        return edge_y
        # edge_y = []
        # for node_i, node_j in edge_index:
        #     node_i_type = y[node_i]
        #     node_j_type = y[node_j]
        #     edge_i_j_type = get_edge_y_category(node_i_type, node_j_type)
        #     edge_y.append(edge_i_j_type)
        # return edge_y
                
    def get_train_dataloader(self, batch_size:int=1, shuffle:bool=True) -> DataLoader:
        '''
            返回训练集DataLoader
            batch_size: 批大小
            shuffle: 是否打乱
        '''
        dataset = self.__datasets[:50]
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_test_dataloader(self, batch_size:int=1, shuffle:bool=True) -> DataLoader:
        '''
            返回测试集DataLoader
            batch_size: 批大小
            shuffle: 是否打乱
        '''
        dataset = self.__datasets[50:]
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
import LLMEnG.src.LLMEnG.base.Parser as Parser
if __name__ == '__main__':
    args = Parser.args
    data_center = DataCenter(args.datasets_json, args.vocab_dir, args.vocab_len, args.embedding_size)
    tarin_dataloader = data_center.get_train_dataloader(args.batch_size, args.shuffle)
    test_dataloader = data_center.get_test_dataloader(args.batch_size, args.shuffle)
    for batch_data in test_dataloader:
            unique_batch_indices = torch.unique(batch_data.batch)
            for batch_index in unique_batch_indices:
                subgraph = batch_data.get_example(batch_index)
                print(subgraph.x.shape)
    
