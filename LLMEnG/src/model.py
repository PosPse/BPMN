import Parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

class GCN(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, node_num_classes=6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(embedding_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, node_num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x
    
class GraphSage(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, node_num_classes=6, aggr='mean'):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(embedding_size, hidden_size, aggr)
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggr)
        self.conv3 = SAGEConv(hidden_size, node_num_classes, aggr)

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
    
class NodeFusion(nn.Module):
    def __init__(self, fusion_method='concat'):
        super(NodeFusion, self).__init__()
        self.fusion_method = fusion_method
        self.linear_src = nn.Linear(64, 64) 
        self.linear_dst = nn.Linear(64, 64)
    def forward(self, src_node, dst_node):
        if self.fusion_method == 'concat':
            return self.concat(src_node, dst_node)

    def concat(self, src_node, dst_node):
        src_node = self.linear_src(src_node)
        dst_node = self.linear_dst(dst_node)
        return torch.cat((src_node, dst_node), dim=1)

class EdgeClassification(nn.Module):
    def __init__(self, embedding_size, edge_num_classes=21):
        super(EdgeClassification, self).__init__()
        self.linear = nn.Linear(embedding_size, edge_num_classes)
        

