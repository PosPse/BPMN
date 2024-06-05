from torch.nn import Embedding
import torch
class Tokenizer():
    def __init__(self, vocab_dir:str, vocab_len:int, embedding_size:int) -> None:
        self.__vocab_dir = vocab_dir
        self.__vocab_len = vocab_len
        self.__embedding_size = embedding_size
        self.__embedding = Embedding(self.__vocab_len, self.__embedding_size)
        self.__vocab_list = None
        self.__init_params()
        # self.vocab = None
        # self.vocab_size = None
        # self.token2id = None
        # self.id2token = None

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
    
    def __id2embedding(self, tokens_id:list) -> list[torch.FloatTensor]:
        '''
            将tokens_id转换为embedding
            tokens_id: list, tokens_id列表
            return: list, tokens_id对应的embedding列表
        '''
        tokens_id = torch.LongTensor(tokens_id)
        tokens_embedding = self.__embedding(tokens_id)
        return tokens_embedding
    
    def token2embedding(self, tokens:list[str]) -> list[torch.FloatTensor]:
        '''
            将tokens转换为embedding
            tokens: list, tokens列表
            return: list, tokens对应的embedding列表
        '''
        tokens_id = self.__token2id(tokens)
        tokens_embedding = self.__id2embedding(tokens_id)
        return tokens_embedding


import Parser
if __name__ == '__main__':
    args = Parser.args
    tokenizer = Tokenizer(args.vocab_dir, args.vocab_len, args.embedding_size)
    tokens_embedding = tokenizer.token2embedding(['hello', 'world'])
    print(tokens_embedding)
    