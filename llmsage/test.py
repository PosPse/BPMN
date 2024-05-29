import torch
from torch_geometric.data import Data

class mydata(Data):
    def __init__(self, x, edge_index, edge_attr=None, y=None, pos=None, time=None, **kwargs):
        super(mydata, self).__init__(x, edge_index, y, edge_attr, pos, time, **kwargs)

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=torch.float)
dataset_cora = mydata(x=x, edge_index=edge_index)
print(dataset_cora)