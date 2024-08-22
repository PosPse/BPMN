import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Linear(feature_dim, 1)

    def forward(self, input_features):
        num_features, batch_size, feature_dim = input_features.shape
        # 计算注意力权重
        attention_weights = torch.tanh(self.attention_layer(input_features.view(-1, feature_dim)))
        attention_weights = nn.functional.softmax(attention_weights.view(num_features, batch_size), dim=0)
        # 融合特征
        fused_features = torch.sum(input_features * attention_weights.unsqueeze(-1), dim=0)
        return fused_features
    
    # 假设输入特征形状为 (num_features, batch_size, feature_dim)
num_features = 3
batch_size = 4
feature_dim = 5
input_features = torch.randn(num_features, batch_size, feature_dim)

attention_fusion_module = AttentionFusion(feature_dim)
fused_feature = attention_fusion_module(input_features)
print(fused_feature.shape)