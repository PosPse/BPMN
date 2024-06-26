import Parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

class GCN(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, num_classes=6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(embedding_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x
    
class GraphSage(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, num_classes=6, aggr='mean'):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(embedding_size, hidden_size, aggr)
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggr)
        self.conv3 = SAGEConv(hidden_size, num_classes, aggr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) 
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

class EdgeClassification(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, num_classes=6):
        super(EdgeClassification, self).__init__()
        

