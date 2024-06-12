from torch.nn import Embedding
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
        位置编码，后期可扩展使用可学习的位置编码
        当前该方法不支持批量处理，即batch_size=1
        PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        
    '''
    def __init__(self, embedding_size:int, max_seq_len:int=512) -> None:
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

        pe = torch.zeros(self.max_seq_len, self.embedding_size)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:list[list[torch.FloatTensor]]) -> list[list[torch.FloatTensor]]:
        x = x + self.pe[:x.size(0)]
        return x

class Tokenizer():
    def __init__(self, vocab_dir:str, vocab_len:int, embedding_size:int) -> None:
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
        # 读取vocab文件为list，去除'\n'
        with open(self.__vocab_dir, 'r', encoding='utf-8') as f:
            vocab_list = f.readlines()
            vocab_list = list(map(lambda token: token.strip(), vocab_list))
            self.__vocab_list = vocab_list
            

    def __token2id(self, tokens:list[str]) -> list[int]:
        '''
            将tokens转换为id
            tokens: list, tokens列表
            return: list, tokens对应的id列表
        '''
        # token统一为小写字母
        tokens = list(map(lambda token: token.casefold(), tokens))
        # 获取token在vocab_list中的id，如果不在vocab_list中，返回'[UNK]'的id
        tokens_id = list(map(lambda token: self.__vocab_list.index(token) if token in self.__vocab_list else self.__vocab_list.index('[UNK]'), tokens))
        return tokens_id
    
    def __id2embedding(self, tokens_id:list) -> list[list[torch.FloatTensor]]:
        '''
            将tokens_id转换为embedding
            tokens_id: list, tokens_id列表
            return: list, tokens_id对应的embedding列表
        '''
        tokens_id = torch.LongTensor(tokens_id)
        with torch.no_grad():
            tokens_embedding = self.__embedding(tokens_id)
        return tokens_embedding
    
    def token2embedding(self, tokens:list[str]) -> list[list[torch.FloatTensor]]:
        '''
            将tokens转换为embedding
            tokens: list, tokens列表
            return: list, tokens对应的embedding列表
        '''
        tokens_id = self.__token2id(tokens)
        tokens_embedding = self.__id2embedding(tokens_id)
        tokens_positional_embedding = self.__positional_encoding(tokens_embedding)
        return tokens_positional_embedding


import Parser
if __name__ == '__main__':
    args = Parser.args
    tokenizer = Tokenizer(args.vocab_dir, args.vocab_len, args.embedding_size)
    tokens_embedding = tokenizer.token2embedding(['hello', 'world', '[activity]'])
    print(tokens_embedding)
    