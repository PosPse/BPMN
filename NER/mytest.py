from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF
import pickle
from config import Config as config
import torch.nn.functional as F
import csv
from collections import OrderedDict
from torch.utils import data
import numpy as np
import re

log_soft = F.log_softmax

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.birnn = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction
        
def load_model(config, do_lower_case=True):
    f = open(config.apr_dir + 'tag2idx.pkl', 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())
    model = Bert_CRF.from_pretrained(config.bert_model, num_labels=len(tag2idx))
    checkpoint = torch.load(config.apr_dir + config.model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # global bert_tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=do_lower_case)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model, bert_tokenizer, unique_labels, tag2idx

input_sentences = "I am submitting the job description for consideration and waiting for the approval."

# read the corpus and return them into list of sentences of list of tokens
def corpus_reader(input):
    tokens = []
    tokens.append(re.findall(r'\w+|[^\w\s]', input))
    return tokens

class NER_Dataset(data.Dataset):
    def __init__(self, tag2idx, sentences, tokenizer_path='', do_lower_case=True):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')

        for i, token in enumerate(sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)

        bert_tokens.append('[SEP]')
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
        return token_ids, len(token_ids), orig_to_tok_map, self.sentences[idx]
def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i > 0) for i in ids] for ids in tok_ids]
    LT = torch.LongTensor
    

    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = LT(tok_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]

    org_tok_map = get_element(2)
    sents = get_element(-1)

    return tok_ids, attn_mask, org_tok_map, sents, list(sorted_idx.cpu().numpy())

def generate_test_data(config, tag2idx, bert_tokenizer="bert-base", do_lower_case=True):
    test_sentences = corpus_reader(input_sentences)
    test_dataset = NER_Dataset(tag2idx, test_sentences, tokenizer_path=bert_tokenizer,
                               do_lower_case=do_lower_case)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_ner(): 
    model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
    test_iter = generate_test_data(config, tag2idx, bert_tokenizer=config.bert_model, do_lower_case=True)

    model.eval()
    for i,batch in enumerate(test_iter):
        token_ids, attn_mask, org_tok_map, original_token, sorted_idx = batch
        print(token_ids)
        print(attn_mask)
        print(org_tok_map)
        print(original_token)
        print(sorted_idx)
        # attn_mask.dt
        inputs = {'input_ids': token_ids.to(device),
                    'attn_masks': attn_mask.to(device)
                }
        with torch.torch.no_grad():
                tag_seqs = model(**inputs)
        
        o2m = org_tok_map[0]
        pos = sorted_idx.index(0)
        for j, orig_tok_idx in enumerate(o2m):
            print(original_token[0][j] + '\t')
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            print(pred_tag + '\n')

test_ner()