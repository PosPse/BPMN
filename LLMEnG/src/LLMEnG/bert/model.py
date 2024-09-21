import Parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

class GCN(nn.Module):
    def __init__(self, hidden_size, node_num_classes=5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(768, 1536)
        self.conv2 = GCNConv(1536, 768)
        self.conv3 = GCNConv(768, 384)
        # self.conv3 = nn.Linear(768, 384)
        self.conv4 = GCNConv(384, hidden_size)
        # self.conv4 = nn.Linear(384, hidden_size)
        self.conv5 = GCNConv(hidden_size, node_num_classes)
        # self.conv5 = nn.Linear(hidden_size, node_num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        # x = self.conv4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv5(x, edge_index)
        # out = self.conv5(x)
        return x, out
    
class GraphSage(nn.Module):
    def __init__(self, hidden_size, node_num_classes=5, aggr='mean'):
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
    def __init__(self, hidden_size, node_num_classes=5, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(768, 384, heads)
        self.conv2 = GATConv(384*heads, 256, heads)
        self.conv3 = GATConv(256*heads, hidden_size, heads)
        self.conv4 = nn.Linear(hidden_size*heads, hidden_size)
        self.conv5 = nn.Linear(hidden_size, node_num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        out = self.conv5(x)
        return x, out

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        num_features, batch_size, feature_dim = x.shape

        attention_weights = self.attention_layer(x.view(-1, feature_dim))
        attention_weights = torch.tanh(attention_weights)
        attention_weights = F.softmax(attention_weights.view(num_features, batch_size), dim=0)

        return torch.sum(x * attention_weights.unsqueeze(-1), dim=0)
    
class FeatureFusionAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureFusionAttention, self).__init__()
        self.feature_dim = feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (num_features, batch_size, feature_dim)
        batch_size = x.size(1)
        
        x = x.permute(1, 0, 2)  # (batch_size, num_features, feature_dim)

        query = self.query(x)  
        key = self.key(x)      
        value = self.value(x)  

        attention_scores = torch.bmm(query, key.permute(0, 2, 1))  
        attention_weights = self.softmax(attention_scores)  

        fused_features = torch.bmm(attention_weights, value)  

        # 将融合特征聚合成 (batch_size, feature_dim)
        output = torch.mean(fused_features, dim=1)  # 或使用 torch.max等其他方法

        return output
        
    
class EdgeFusion(nn.Module):
    def __init__(self, hidden_size, fusion_method='concat'):
        super(EdgeFusion, self).__init__()
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method
        self.src_node_type_linear1 = nn.Linear(768, 384)
        self.src_node_type_linear2 = nn.Linear(384, hidden_size)
        self.dst_node_type_linear1 = nn.Linear(768, 384)
        self.dst_node_type_linear2 = nn.Linear(384, hidden_size)
        self.attention_layer = FeatureFusionAttention(hidden_size)

    def forward(self, src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb):
        if self.fusion_method == 'concat':
            return self.concat(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)
        elif self.fusion_method == 'attention':
            return self.attention(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)

    def concat(self, src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb):
        src_node_type_emb = self.src_node_type_linear1(src_node_type_emb)
        src_node_type_emb = F.relu(src_node_type_emb)
        src_node_type_emb = F.dropout(src_node_type_emb, training=self.training)
        src_node_type_emb = self.src_node_type_linear2(src_node_type_emb)

        dst_node_type_emb = self.dst_node_type_linear1(dst_node_type_emb)
        dst_node_type_emb = F.relu(dst_node_type_emb)
        dst_node_type_emb = F.dropout(dst_node_type_emb, training=self.training)
        dst_node_type_emb = self.dst_node_type_linear2(dst_node_type_emb)
        # src_node_emb = self.src_node_type_linear1(src_node_emb)
        # src_node_emb = F.relu(src_node_emb)
        # src_node_emb = F.dropout(src_node_emb, training=self.training)
        # src_node_emb = self.src_node_type_linear2(src_node_emb)

        # dst_node_emb = self.dst_node_type_linear1(dst_node_emb)
        # dst_node_emb = F.relu(dst_node_emb)
        # dst_node_emb = F.dropout(dst_node_emb, training=self.training)
        # dst_node_emb = self.dst_node_type_linear2(dst_node_emb)
        

        # with torch.no_grad():
        return torch.cat((src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb), dim=1)
        # return torch.cat((src_node_emb, dst_node_emb, src_2_dst_distance_emb), dim=1)
    
    def attention(self, src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb):
        src_node_type_emb = self.src_node_type_linear1(src_node_type_emb)
        src_node_type_emb = F.relu(src_node_type_emb)
        src_node_type_emb = F.dropout(src_node_type_emb, training=self.training)
        src_node_type_emb = self.src_node_type_linear2(src_node_type_emb)

        dst_node_type_emb = self.dst_node_type_linear1(dst_node_type_emb)
        dst_node_type_emb = F.relu(dst_node_type_emb)
        dst_node_type_emb = F.dropout(dst_node_type_emb, training=self.training)
        dst_node_type_emb = self.dst_node_type_linear2(dst_node_type_emb)

        edge_emb = torch.stack([src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb], dim=0)
        return self.attention_layer(edge_emb)

class EdgeClassification(nn.Module):
    def __init__(self, device, hidden_size, edge_fusion, edge_num_classes=5):
        super(EdgeClassification, self).__init__()
        self.device = device
        self.hidden_size = hidden_size * 5
        # self.hidden_size = hidden_size
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
    def forward(self, node_embedding, subgraph, node_type_pred_emb):
        need_pred_edge = subgraph.need_pred_edge
        src_node_emb = []
        dst_node_emb = []
        src_node_type_emb = []
        dst_node_type_emb = []
        src_2_dst_distance = []
        for node_i, node_j in need_pred_edge:
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
        src_2_dst_distance_emb = self.distance_MinMaxScaler_emb(src_2_dst_distance)

        fused_node = self.edge_fusion(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)
        edge_feature = self.linear1(fused_node)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear2(edge_feature)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear3(edge_feature)
        return edge_feature
    def forward1(self, node_embedding, subgraph, node_type_pred_emb):
        src_node_emb = []
        dst_node_emb = []
        src_node_type_emb = []
        dst_node_type_emb = []
        src_2_dst_distance = []
        for node_i in range(0, subgraph.num_nodes): 
            for node_j in range(node_i+1, subgraph.num_nodes):
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
        src_2_dst_distance_emb = self.distance_MinMaxScaler_emb(src_2_dst_distance)

        fused_node = self.edge_fusion(src_node_emb, dst_node_emb, src_node_type_emb, dst_node_type_emb, src_2_dst_distance_emb)
        edge_feature = self.linear1(fused_node)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear2(edge_feature)
        edge_feature = F.relu(edge_feature)
        edge_feature = F.dropout(edge_feature, training=self.training)
        edge_feature = self.linear3(edge_feature)
        return edge_feature


        
        

