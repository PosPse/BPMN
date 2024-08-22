import Parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

class GCN(nn.Module):
    def __init__(self, hidden_size, node_num_classes=6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(768, 1536)
        self.conv2 = GCNConv(1536, 768)
        self.conv3 = GCNConv(768, 384)
        self.conv4 = GCNConv(384, hidden_size)
        self.conv5 = GCNConv(hidden_size, node_num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv5(x, edge_index)
        return x, out
    
class GraphSage(nn.Module):
    def __init__(self, hidden_size, node_num_classes=6, aggr='mean'):
        super(GraphSage, self).__init__()
        # self.conv1 = SAGEConv(768, hidden_size, aggr)
        # self.conv2 = SAGEConv(hidden_size, hidden_size, aggr)
        # self.conv3 = SAGEConv(hidden_size, node_num_classes, aggr)
        self.conv1 = SAGEConv(768, 1536, aggr)
        self.conv3 = SAGEConv(1536, 768, aggr)
        self.conv4 = SAGEConv(768, 384, aggr)
        self.conv5 = SAGEConv(384, hidden_size, aggr)
        self.conv6 = SAGEConv(hidden_size, node_num_classes, aggr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index) 
        # x = F.dropout(x, training=self.training)
        # x = F.relu(x)
        # out = self.conv3(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv6(x, edge_index)
        return x, out
    
class GAT(nn.Module):
    def __init__(self, hidden_size, node_num_classes=6, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(768, hidden_size, heads)
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
    def __init__(self, hidden_size, fusion_method='concat'):
        super(EdgeFusion, self).__init__()
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method
        self.src_node_type_linear1 = nn.Linear(768, 384)
        self.src_node_type_linear2 = nn.Linear(384, hidden_size)
        self.dst_node_type_linear1 = nn.Linear(768, 384)
        self.dst_node_type_linear2 = nn.Linear(384, hidden_size)

    def forward(self, src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb):
        if self.fusion_method == 'concat':
            return self.concat(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)

    def concat(self, src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb):
        src_node_type_emb = self.src_node_type_linear1(src_node_type_emb)
        src_node_type_emb = F.relu(src_node_type_emb)
        src_node_type_emb = F.dropout(src_node_type_emb, training=self.training)
        src_node_type_emb = self.src_node_type_linear2(src_node_type_emb)

        dst_node_type_emb = self.dst_node_type_linear1(dst_node_type_emb)
        dst_node_type_emb = F.relu(dst_node_type_emb)
        dst_node_type_emb = F.dropout(dst_node_type_emb, training=self.training)
        dst_node_type_emb = self.dst_node_type_linear2(dst_node_type_emb)

        # with torch.no_grad():
        return torch.cat((src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb), dim=1)

class EdgeClassification(nn.Module):
    def __init__(self, device, hidden_size, edge_fusion, edge_num_classes=10):
        super(EdgeClassification, self).__init__()
        self.device = device
        self.hidden_size = hidden_size * 5
        self.edge_fusion = edge_fusion
        self.edge_num_classes = edge_num_classes
        self.linear1 = nn.Linear(self.hidden_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, edge_num_classes)
        
    def distance_MinMaxScaler_emb(self, src_2_dst_distance:list[int]):
        src_2_dst_distance = torch.tensor(src_2_dst_distance, dtype=torch.float32)
        min_distance = src_2_dst_distance.min()
        max_distance = src_2_dst_distance.max()
        distance = (src_2_dst_distance - min_distance) / (max_distance - min_distance)
        distance = distance.unsqueeze(1).repeat(1, 128)
        return distance.to(self.device)
    
    def distance_Z_score_emb(self, src_2_dst_distance:list[int]):
        src_2_dst_distance = torch.tensor(src_2_dst_distance, dtype=torch.float32)
        mean_distance = src_2_dst_distance.mean()
        std_distance = src_2_dst_distance.std()
        distance = (src_2_dst_distance - mean_distance) / std_distance
        distance = distance.unsqueeze(1).repeat(1, 128)
        return distance.to(self.device)
    def forward(self, node_embedding, raw_data, node_type_pred_emb):
        src_node_emb = []
        dst_node_emb = []
        src_node_type_emb = []
        dst_node_type_emb = []
        src_2_dst_distance = []
        for node_i in range(raw_data.num_nodes): 
            for node_j in range(raw_data.num_nodes):
                src_node = node_embedding[node_i]
                dst_node = node_embedding[node_j]
                src_node_type = node_type_pred_emb[node_i]
                dst_node_type = node_type_pred_emb[node_j]
                src_node_emb.append(src_node)
                dst_node_emb.append(dst_node)
                src_node_type_emb.append(src_node_type)
                dst_node_type_emb.append(dst_node_type)
                src_2_dst_distance.append((node_i - node_j) if (node_i - node_j) > 0 else (node_j - node_i))
        src_node_emb = torch.stack(src_node_emb, dim=0)
        dst_node_emb = torch.stack(dst_node_emb, dim=0)
        src_node_type_emb = torch.stack(src_node_type_emb, dim=0)
        dst_node_type_emb = torch.stack(dst_node_type_emb, dim=0)
        src_2_dst_distance_emb = self.distance_Z_score_emb(src_2_dst_distance)

        fused_node = self.edge_fusion(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)
        edge_feature = self.linear1(fused_node)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear2(edge_feature)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear3(edge_feature)
        return edge_feature


        
        

