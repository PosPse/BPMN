import numpy as np
import torch

from util import *
from attention import MultiHeadedAttention
from positionwiseFeedForward import PositionwiseFeedForward
from positional_encoding import PositionalEncoding
from encoder import Encoder, EncoderLayer
from data_read import data_concat_padding
import dataCenter


class Framework(nn.Module):
    def __init__(self, encoder, src_embed, max_len, Data_center):
        """
        标准的encoder-decoder架构
        :param encoder: (nn.Module) transformer编码器模型
        :param src_embed: (nn.Module) 输入词向量（embedding层）
        :param generator: (nn.Module) 成器，实现transformer的encoder最后的linear+softmax
        """
        super(Framework, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.max_len = max_len
        self.data_center = Data_center

    def forward(self, src):
        """
        喂入和处理masked src和目标序列.
        decode的输入是encode输出，目标词向量
        :param src: (Tensor) 输入词向量
        :param tgt: (Tensor) 输出词向量
        :param src_mask:
        :param tgt_mask:
        :return: (nn.Module) 整个transformer模型
        此处已经返回了1*n*3的矩阵了
        """
        return self.encode(src)

    def encode(self, src):
        # src_embed同时搞定了embedding和位置编码,在此处搞定mask
        # src(batch_num(1), max_len, d_model)
        num_pad = self.max_len - src.size()[1]
        src_mask_true = torch.Tensor(np.ones((src.size()[1], self.max_len), dtype=bool))
        src_mask_false = torch.Tensor(np.zeros((num_pad, self.max_len), dtype=bool))
        src_mask = torch.cat([src_mask_true, src_mask_false], 0)
        src_mask = (src_mask == 1)
        src_mask = src_mask.unsqueeze(0).unsqueeze(0)
        pad_id = [0] * num_pad
        pad_embedding = self.data_center.id2embedding(pad_id, is_numpy=False)
        pad_embedding = pad_embedding.unsqueeze(0)
        src = torch.cat([src, pad_embedding], 1)
        return self.encoder(self.src_embed(src), src_mask)

def make_model(Data_center, max_len, batch_num, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建整体模型
    :param N: (int, default=6) 编解码层的重复次数
    :param d_model: (int, default=512) embedding后词向量维度
    :param d_ff: (int, default=2048) 编解码器内层维度
    :param h: (int, default=8) 'Scaled Dot Product Attention'，使用的次数（多头注意力的头数）
    :param dropout: (int, default=0.1) 丢弃机制，正则化的一种方式，默认为0.1
    :return: (nn.Module) 整个transformer模型
    """
    '''copy.deepcopy深复制'''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len=max_len, batch_num=batch_num)
    model = Framework(
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        src_embed = c(position),
        max_len=max_len,
        Data_center = Data_center
    )

    # 下面这部分非常重要，模型参数使用Xavier初始化方式，基本思想是输入和输出的方差相同，包括前向传播和后向传播
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
