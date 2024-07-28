import Parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

class GCN(nn.Module):
    def __init__(self, embedding_size, hidden_size, node_num_classes=6):
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
    def __init__(self, embedding_size=128, hidden_size=128, node_num_classes=6, aggr='mean'):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(embedding_size, hidden_size, aggr)
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggr)
        self.conv3 = SAGEConv(hidden_size, node_num_classes, aggr)

    def forward(self, data, use_last_layer=True):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) 
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        if use_last_layer:
            x = self.conv3(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=64, node_num_classes=6, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(embedding_size, hidden_size, heads)
        self.conv2 = GATConv(hidden_size*heads, node_num_classes, heads=1)
    
    def forward(self, data, use_last_layer=True):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.log_softmax(x, dim=1)
        return x
    
class EdgeFusion(nn.Module):
    def __init__(self, fusion_method='concat'):
        super(EdgeFusion, self).__init__()
        self.fusion_method = fusion_method
        self.linear_src = nn.Linear(128, 64) 
        self.linear_dst = nn.Linear(128, 64)
    def forward(self, src_node, dst_node):
        if self.fusion_method == 'concat':
            return self.concat(src_node, dst_node)

    def concat(self, src_node, dst_node):
        src_node = self.linear_src(src_node)
        dst_node = self.linear_dst(dst_node)
        return torch.cat((src_node, dst_node), dim=1)

class EdgeClassification(nn.Module):
    def __init__(self, node_fusion, edge_num_classes=21):
        super(EdgeClassification, self).__init__()
        self.node_fusion = node_fusion
        self.linear_src = nn.Linear(128, edge_num_classes)
        

    def forward(self, node_embedding, data):
        res = []
        x, edge_index = data.x, data.edge_index
        # for node_i in range(data.num_nodes): 
        #     for node_j in range(data.num_nodes):
        #         src_node = node_embedding[node_i]
        #         src_node = src_node.reshape(1, -1)
        #         dst_node = node_embedding[node_j]
        #         dst_node = dst_node.reshape(1, -1)
        #         fused_node = self.node_fusion(src_node, dst_node)
        #         edge_feature = self.linear_src(fused_node)
        #         res.append(edge_feature)
        # res = torch.cat(res, dim=0)
        for node_i, node_j in edge_index.t().tolist():
            src_node = node_embedding[node_i]
            src_node = src_node.reshape(1, -1)
            dst_node = node_embedding[node_j]
            dst_node = dst_node.reshape(1, -1)
            fused_node = self.node_fusion(src_node, dst_node)
            edge_feature = self.linear_src(fused_node)
            res.append(edge_feature)
        res = torch.cat(res, dim=0)
        return res
                


        
        

