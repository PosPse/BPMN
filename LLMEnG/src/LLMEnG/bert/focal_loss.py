import torch
import torch.nn as nn
import torch.nn.functional as F
import Parser

class FocalLossWithBinaryCrossEntropy(nn.Module):
    def __init__(self, device, alpha, gamma=2, reduction='mean', edge_num_classes=5):
        """
        Focal Loss的构造函数，允许为每个类别设置不同的alpha值。
            :param device: 计算设备，如'cpu'或'cuda'。
            :param alpha: 一个列表或张量，包含每个类别的alpha值。
            :param gamma: 调节损失函数的难易样本权重。
            :param reduction: 指定损失计算方式，'mean'或'sum'。
            :param edge_num_classes: 边的类别数。
        """
        super(FocalLossWithBinaryCrossEntropy, self).__init__()
        self.device = device
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.tensor([1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=torch.float32)).to(device)
        self.gamma = gamma
        self.reduction = reduction
        self.edge_num_classes = edge_num_classes
    
    def forward(self, inputs, targets):
        """
        Focal Loss的前向传播。
            :param inputs: 模型的原始输出，维度为[batch_size, num_classes]。
            :param targets: 真实标签，维度为[batch_size]。
            :return: 计算得到的Focal Loss。
        """
        # 将目标转换为one-hot编码
        one_hot_targets = F.one_hot(targets, num_classes=self.edge_num_classes).float()
        # 计算交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction='none').to(self.device)
        
        # 根据目标调整alpha
        alpha_factor = self.alpha[targets].unsqueeze(1).to(self.device)

        # 计算Focal Loss
        pt = torch.exp(-bce_loss)  # 计算权重

        loss = alpha_factor * ((1 - pt) ** self.gamma) * bce_loss

        # 根据reduction参数处理损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

class FocalLossWithCrossEntropy(nn.Module):
    def __init__(self, device, alpha, gamma=2, reduction='mean'):
        """
        Focal Loss的构造函数，允许为每个类别设置不同的alpha值。
            :param device: 计算设备，如'cpu'或'cuda'。
            :param alpha: 一个列表或张量，包含每个类别的alpha值。
            :param gamma: 调节损失函数的难易样本权重。
            :param reduction: 指定损失计算方式，'mean'或'sum'。
        """
        super(FocalLossWithCrossEntropy, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction=self.reduction).to(self.device)

    def forward(self, inputs, targets):
        """
        Focal Loss的前向传播。
            :param inputs: 模型的原始输出，维度为[batch_size, num_classes]。
            :param targets: 真实标签，维度为[batch_size]。
            :return: 计算得到的Focal Loss。
        """
        # 计算交叉熵损失
        ce_loss = self.cross_entropy_loss(inputs, targets)

        # 计算概率
        pt = torch.exp(-ce_loss) 

        # 计算Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

if __name__ == '__main__':
    args = Parser.args
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    inputs = torch.randn(5, args.edge_num_classes).to(device)  # 假设batch_size=10，num_classes=10
    targets = torch.tensor([1, 0, 4, 1, 3], dtype=torch.long).to(device)   # 类别
    alpha_values = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32).to(device)  # 类别对应的alpha值

    # 实例化FocalLoss
    # focal_loss = FocalLossWithBinaryCrossEntropy(device=device, alpha=alpha_values, gamma=2, reduction='mean')
    focal_loss = FocalLossWithCrossEntropy(device=device, alpha=alpha_values, gamma=2, reduction='mean')
    # 计算损失
    loss = focal_loss(inputs, targets)
    print(loss)