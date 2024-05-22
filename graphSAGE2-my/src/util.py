import copy
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
#from torchviz import make_dot
#from data_read import draw
from data_read import assess

def clones(module, N):
    """
    生成N个相同的层
    :param module:(nn.Module)输入模型
    :param N:(int)重复次数
    :return: 复制生成的模型列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        归一化，即每个子层的输出为LayerNorm(x+Sublayer(x)),(x+Sublayer(x)是子层自己实现的功能。
        将 dropout 应用于每个子层的输出，然后再将其添加到子层输入中并进行归一化。
        为了促进这些残差连接，模型中的所有子层以及嵌入层产生维度输出为512
        :param features:
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        """
        残差连接模块，对应论文的 Add & Norm
        :param size: (int)模型尺寸
        :param dropout: (int)丢弃机制
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播，将输入与正则化的输出相加
        :param x:
        :param sublayer:
        :return:
        """
        return x + self.dropout(sublayer(self.norm(x)))


def run_epoch(data_iter, model, loss_compute, concat_max_len, mode):
    """
    通用的训练和评分函数来跟踪损失。传入一个通用的损失计算函数处理参数更新。
    :param data_iter:
    :param model:
    :param loss_compute:
    :return:
    """
    total_loss = 0
    total_batch = 0
    for i, batch in enumerate(data_iter):
        total_batch += 1
        if torch.cuda.is_available():
            batch.src = batch.src.cuda()
            batch.src_mask = batch.src_mask.cuda()
        out = model.forward(batch.src, batch.src_mask, batch.activity_num, concat_max_len, batch.activity_index)
        # 1.放到generator里面输出结果， 2.计算loss 3.梯度下降
        if mode == 'train':
            loss = loss_compute(out, batch.trg)
        elif mode == 'test':
            loss, class_output = loss_compute(out, batch.trg)
            #draw(batch.file_name[0], tag_seq=class_output, activity_num=batch.activity_num[0])
        total_loss += loss
    return total_loss / total_batch


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        优化器：论文用的是adam，这个类主要用于针对不同模型尺寸动态更新学习率
        :param model_size:
        :param factor:
        :param warmup:
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # 更新参数和学习率
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 执行上面更新的学习率
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """
    优化器调用示例：
    :param model:
    :return:
    """
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, mode):
        """
        标签平滑:论文正则化的一种方式，另外就是使用dropout了
        在训练期间，使用values的标签平滑，使用 KL div 损失实现标签平滑，防止模型过度自信预测
        论文没有使用 one-hot 目标分布，而是创建了一个分布，该分布具有confidence正确的单词和分布在整个词汇表中的其余smoothing。
        :param size: (int) 模型尺寸，对应词向量长度
        :param padding_idx: (int) 填充步幅
        :param smoothing:
        """
        super(LabelSmoothing, self).__init__()
        self.mode = mode

    def forward(self, x, label):
        class_output = []
        cross_entropy = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.23954983922, 5.3172413793, 192.75]))
        if torch.cuda.is_available():
            cross_entropy = cross_entropy.cuda()
        # 如果是测试，输出分类结果
        if self.mode == 'test':
            x = x.squeeze(0)
            padding_index = -1
            for j in label[0]:
                if j.equal(torch.LongTensor([0, 0, 0])):
                    break
                else:
                    padding_index += 1
            for i in range(0,padding_index+1):
                max_index = 0
                if x[i][1] >= x[i][max_index]:
                    max_index = 1
                if x[i][2] >= x[i][max_index]:
                    max_index = 2
                if max_index == 0:
                    class_output.append(0)
                if max_index == 1:
                    class_output.append(1)
                if max_index == 2:
                    class_output.append(2)
            x = x.unsqueeze(0)
            assess(x[0][:padding_index+1],label[0][:padding_index+1])
        for i in range(0, len(x)):
            padding_index = -1
            for j in label[i]:
               if j.equal(torch.LongTensor([0,0,0])):
                   break
               else:
                   padding_index += 1
            if torch.cuda.is_available():
                temp_loss = cross_entropy(x[i][0:padding_index+1].cuda(), label[i][0:padding_index+1].float().cuda())
            else:
                temp_loss = cross_entropy(x[i][0:padding_index + 1], label[i][0:padding_index + 1].float())
            if i == 0:
                loss = temp_loss
            else:
                loss += temp_loss
        return loss, class_output




