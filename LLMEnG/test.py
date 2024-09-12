# # import torch
# # import torch.nn as nn

# # class AttentionFusion(nn.Module):
# #     def __init__(self, feature_dim):
# #         super(AttentionFusion, self).__init__()
# #         self.attention_layer = nn.Linear(feature_dim, 1)

# #     def forward(self, input_features):
# #         num_features, batch_size, feature_dim = input_features.shape
# #         # 计算注意力权重
# #         attention_weights = torch.tanh(self.attention_layer(input_features.view(-1, feature_dim)))
# #         attention_weights = nn.functional.softmax(attention_weights.view(num_features, batch_size), dim=0)
# #         # 融合特征
# #         fused_features = torch.sum(input_features * attention_weights.unsqueeze(-1), dim=0)
# #         return fused_features
    
# #     # 假设输入特征形状为 (num_features, batch_size, feature_dim)
# # num_features = 3
# # batch_size = 20
# # feature_dim = 128
# # input_features = torch.randn(num_features, batch_size, feature_dim)
# # print(input_features.shape)

# # attention_fusion_module = AttentionFusion(feature_dim)
# # fused_feature = attention_fusion_module(input_features)
# # print(fused_feature.shape)

# import torch
# import torch.nn as nn

# class FeatureFusionAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super(FeatureFusionAttention, self).__init__()
#         self.feature_dim = feature_dim
        
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # x shape: (num_features, batch_size, feature_dim)
#         batch_size = x.size(1)
        
#         x = x.permute(1, 0, 2)  # (batch_size, num_features, feature_dim)

#         query = self.query(x)  
#         key = self.key(x)      
#         value = self.value(x)  

#         attention_scores = torch.bmm(query, key.permute(0, 2, 1))  
#         attention_weights = self.softmax(attention_scores)  

#         fused_features = torch.bmm(attention_weights, value)  

#         # 将融合特征聚合成 (batch_size, feature_dim)
#         output = torch.mean(fused_features, dim=1)  # 或使用 torch.max等其他方法

#         return output

# # 示例参数
# num_features = 5
# batch_size = 10
# feature_dim = 128

# # 创建输入 tensor
# input_tensor = torch.rand(num_features, batch_size, feature_dim)

# # 实例化模型并前向传播
# model = FeatureFusionAttention(feature_dim)
# output = model(input_tensor)

# # 输出结果
# print(output.shape)  # 应该是 (batch_size, feature_dim)
print(111)
a = [1,3,4]