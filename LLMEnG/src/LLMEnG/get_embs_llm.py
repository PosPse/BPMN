import sys
sys.path.append('/home/btr/bpmn/LLMEnG/src')
from transformers import BertTokenizer, BertModel
import torch
from model_config import ModelConfig

class Tokenizer():
    def __init__(self, device:torch.device, the_way_of_emb_new_token:str = 'mean', the_way_of_token_emb:str = 'cls', the_way_of_fussion_node_emb:str = 'sum', node_emb_alpha:float = 1.0, node_type_emb_beta:float = 0.5) -> None:
        '''
            初始化Tokenizer，获取每个样本（一个图对应的文本）的词向量
            device: str, GPU设备
            the_way_of_emb_new_token: str, 新token的嵌入方式
            the_way_of_token_emb: str, token的嵌入方式
            the_way_of_fussion_node_emb: str, 节点嵌入的融合方式
            node_emb_alpha: float, 节点嵌入中，节点本身嵌入的权重(1-node_emb_alpha), 节点类型嵌入的权重(node_emb_alpha)
        '''
        self.__llm_model = ModelConfig.get_current_model().model_path
        self.__device = device
        self.__the_way_of_emb_new_token = the_way_of_emb_new_token
        self.__the_way_of_token_emb = the_way_of_token_emb
        self.__the_way_of_fussion_node_emb = the_way_of_fussion_node_emb
        self.__node_emb_alpha = node_emb_alpha
        self.tokenizer = BertTokenizer.from_pretrained(self.__llm_model)
        self.model = BertModel.from_pretrained(self.__llm_model).to(self.__device)
        self.embedding_size = self.model.config.hidden_size
        self.__init_params()

    def __init_params(self) -> None:
        new_token = {'[activity]': "activity event entity",
                     '[condition]': "gateway conditions",
                     '[sign-successor]': "sequential signal words",
                     '[sign-selection]': "selective signal words",
                     '[sign-parallel]': "parallel signal words",
                     '[sign-loop]': "cyclic signal words"}
        # 添加新token
        self.tokenizer.add_tokens(list(new_token.keys()), special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.__new_token_dict = new_token 
        self.__new_token_size = len(new_token)
        # 为新token添加嵌入
        self.__init_new_token_emb()

    def __init_new_token_emb(self):
        '''
            为新token添加嵌入
        '''
        if self.__the_way_of_emb_new_token == 'random':
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif self.__the_way_of_emb_new_token == 'zero':
            with torch.no_grad():
                # self.model.embeddings.word_embeddings.weight[-self.__new_token_size:].fill_(0)
                self.model.embeddings.word_embeddings.weight[-self.__new_token_size:, :] = torch.zeros([self.__new_token_size, self.model.config.hidden_size], requires_grad=True)
        elif self.__the_way_of_emb_new_token == 'copy':
            pass
        elif self.__the_way_of_emb_new_token == 'mean':
            with torch.no_grad():
                for i, (k, v) in enumerate(reversed(self.__new_token_dict.items()), start=1):
                    tokenized = self.tokenizer.tokenize(v)
                    tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
                    new_token_emb = self.model.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                    self.model.embeddings.word_embeddings.weight[-i, :] = new_token_emb.clone().detach().requires_grad_(True)
        # print(self.model.embeddings.word_embeddings.weight[-6:, :])

    def __node_emb_fussion(self, node_emb:list[torch.FloatTensor], node_type_emb:list[torch.FloatTensor]):
        '''
            节点嵌入融合
            node_emb: 节点嵌入
            node_type_emb: 节点类型嵌入
        '''
        if self.__the_way_of_fussion_node_emb == 'sum':
            return node_emb * (1- self.__node_emb_alpha) + node_type_emb * self.__node_emb_alpha
        elif self.__the_way_of_fussion_node_emb == 'auto':
            pass
        elif self.__the_way_of_fussion_node_emb == 'concat':
            pass
    
    def __token_emb(self, node_token:list[str], node_token_type:list[str]):
        '''
            获取token的嵌入
            token: token列表
            token_type: token类型列表
        '''
        node_token_emb = None
        node_token_type_emb = None
        if self.__the_way_of_token_emb == 'cls':
            with torch.no_grad():
                inputs_token = self.tokenizer(node_token, return_tensors='pt', padding=True, truncation=True).to(self.__device)
                inputs_token_type = self.tokenizer(node_token_type, return_tensors='pt', padding=True, truncation=True).to(self.__device)
                outputs_token = self.model(**inputs_token)
                outputs_token_type = self.model(**inputs_token_type)
                node_token_emb = outputs_token.last_hidden_state[:, 0, :]
                node_token_type_emb = outputs_token_type.last_hidden_state[:, 0, :] 
        elif self.__the_way_of_token_emb == 'mean':
            pass
        elif self.__the_way_of_token_emb == 'pooler_output':
            pass
        return node_token_emb, node_token_type_emb

    def node2embedding(self, node_token:list[str], node_token_type:list[str]):
        node_emb, node_type_emb = self.__token_emb(node_token, node_token_type)
        return self.__node_emb_fussion(node_emb, node_type_emb)

import Parser
if __name__ == '__main__':
    args = Parser.args
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(device=device)
    ModelConfig.set_current_model(args.model_name)
    print(ModelConfig.get_current_model().model_name)
    node_token_list = [
            "The credit company receives the credit information from the customer",
            "If",
            "the requested amount is greater than 1M $",
            "an approval must be requested",
            "If",
            "the requested amount is lower or equal to 1M $",
            "the company assess the risk of the credit",
            "After",
            "the assessment",
            "if",
            "the risk is high",
            "an approval must be requested",
            "but if",
            "the risk is low",
            "the credit is accepted",
            "After",
            "the approval request",
            "the credit could be accepted or rejected",
            "in both cases",
            "an email is sent to the customer"
        ]
    data_2_mask_single_signal_llm = [
            "[activity]",
            "[sign-selection]",
            "[condition]",
            "[activity]",
            "[sign-selection]",
            "[condition]",
            "[activity]",
            "[sign-successor]",
            "[condition]",
            "[sign-selection]",
            "[condition]",
            "[activity]",
            "[sign-selection]",
            "[condition]",
            "[activity]",
            "[sign-successor]",
            "[condition]",
            "[activity]",
            "[sign-selection]",
            "[activity]"
        ]
    print(tokenizer.node2embedding(node_token_list, data_2_mask_single_signal_llm).shape)


    
    
    