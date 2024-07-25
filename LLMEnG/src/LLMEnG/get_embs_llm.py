from torch.nn import Embedding
import torch
import torch.nn as nn

class Tokenizer():
    def __init__(self, llm_model:str) -> None:
        '''
            初始化Tokenizer，获取每个样本（一个图对应的文本）的词向量
            vocab_dir: str, 词表文件路径
            vocab_len: int, 词表长度
            embedding_size: int, 词向量维度
        '''
        self.__vocab_dir = vocab_dir
        self.__vocab_len = vocab_len
        self.__embedding_size = embedding_size
        self.__embedding = Embedding(self.__vocab_len, self.__embedding_size)
        self.__positional_encoding = PositionalEncoding(self.__embedding_size)
        self.__vocab_list = None
        self.__init_params()

    def __init_params(self) -> None:
        pass

import Parser
if __name__ == '__main__':
    args = Parser.args
    
    