#!/usr/bin/env python
# -*- encoding: utf-8
'''
@file: ner.py
@time: @Time : 4/26/21 2:42 PM 
@desc：
'''
from io import open
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from transformers import BertTokenizer
import numpy as np
import os
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF
import timeit
import subprocess
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
import datetime
from config import Config as config
import spacy
from optparse import OptionParser
import sys
from collections import OrderedDict
import  csv

def check(token):
    for i in token:
        if i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return False
        if i in [':',';','.','\'','\"','-','%','$','(',')',',']:
            return False
        return True
# to initialize the network weight with fix seed.
def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# read the corpus and return them into list of sentences of list of tokens
'''
tokens: [['Depending', 'on', 'the', 'customers', 'acceptance', '/', 'rejection', 'the', 'process', 'flow', 'at', 'customer', 'service', 'either', 'ends', '(', 'in', 'case', 'of', 'withdrawal', ')', 'or', 'continues', '(', 'in', 'case', 'of', 'a', 'confirmation', ')', '.']...]
labels: [['B-signal', 'I-signal', 'B-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'B-activity', 'I-activity', 'I-activity', 'I-activity', 'I-activity', 'I-activity', 'I-activity', 'I-activity', 'O', 'B-signal', 'I-signal', 'I-signal', 'B-condition', 'O', 'B-signal', 'B-activity', 'O', 'B-signal', 'I-signal', 'I-signal', 'B-condition', 'I-condition', 'O', 'punctuation']...]
label_set: ['B-signal', 'I-signal', 'B-condition', 'I-condition', 'B-activity', 'I-activity', 'O', 'punctuation']
'''
def corpus_reader(path, word_idx=0, label_idx=1):
    tokens, labels = [], []
    tmp_tok, tmp_lab = [], []
    label_set = []
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    #number = 1
    for line in reader:
        #print(str(number) + line[0] + 'begin')
        if line == []:
            if len(tmp_tok) > 0:
                tokens.append(tmp_tok)
                labels.append(tmp_lab)
            tmp_tok = []
            tmp_lab = []
        else:
            #if check(line[0]):
                #print('number:'+str(number)+'  '+'token:'+line[word_idx]+'   label:'+line[label_idx])
            tmp_tok.append(line[word_idx])
            tmp_lab.append(line[label_idx])
            label_set.append(line[label_idx])
        #print(str(number) + line[0] + 'end')
        #number += 1
    return tokens, labels, list(OrderedDict.fromkeys(label_set))

def corpus_reader_use(path, delim='\t', word_idx=0):
    tokens = []
    tmp_tok = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            if cols == [''] or cols == ['[lwx]']:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok)
                tmp_tok = []
            else:
                tmp_tok.append(cols[word_idx])
    return tokens

class NER_Dataset(data.Dataset):
    def __init__(self, tag2idx, sentences, labels, tokenizer_path='', do_lower_case=True):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = []
        for x in self.labels[idx]:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        # append dummy label 'X' for subtokens
        modified_labels = [self.tag2idx['X']]
        for i, token in enumerate(sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            modified_labels.append(label[i])
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
            modified_labels.extend([self.tag2idx['X']] * (len(new_token) - 1))

        bert_tokens.append('[SEP]')
        modified_labels.append(self.tag2idx['X'])
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
            modified_labels = modified_labels[:512]
        return token_ids, len(token_ids), orig_to_tok_map, modified_labels, self.sentences[idx]

def get_use_data(sentences, idx, tokenizer, do_lower_case=True):

    sentence = sentences[idx]
    bert_tokens = []
    orig_to_tok_map = []
    bert_tokens.append('[CLS]')
    for i, token in enumerate(sentence):
        if len(bert_tokens) >= 512:
            print("句子长度溢出警告")
            break
        orig_to_tok_map.append(len(bert_tokens))
        new_token = tokenizer.tokenize(token)
        bert_tokens.extend(new_token)

    bert_tokens.append('[SEP]')
    token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    if len(token_ids) > 511:
        token_ids = token_ids[:512]
    return token_ids, len(token_ids), orig_to_tok_map, sentences[idx]

def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i > 0) for i in ids] for ids in tok_ids]
    LT = torch.LongTensor
    label = do_pad(3, maxlen)

    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = LT(tok_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    labels = LT(label)[sorted_idx]
    org_tok_map = get_element(2)
    sents = get_element(-1)

    return tok_ids, attn_mask, org_tok_map, labels, sents, list(sorted_idx.cpu().numpy())

def pad_use(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i > 0) for i in ids] for ids in tok_ids]
    org_tok_map = get_element(2)
    sents = get_element(-1)

    return tok_ids, attn_mask, org_tok_map, sents

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction

'''
获取训练数据：DataLoader(train_iter, eval_iter, tag2idx)
'''
def generate_training_data(config, bert_tokenizer="bert-base", do_lower_case=True):
    training_data, validation_data = config.data_dir + config.training_data, config.data_dir + config.val_data
    train_sentences, train_labels, label_set = corpus_reader(training_data)
    label_set.append('X')
    tag2idx = {t: i for i, t in enumerate(label_set)}
    # print('Training datas: ', len(train_sentences))
    train_dataset = NER_Dataset(tag2idx, train_sentences, train_labels, tokenizer_path=bert_tokenizer,
                                do_lower_case=do_lower_case)
    # save the tag2indx dictionary. Will be used while prediction
    with open(config.apr_dir + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    dev_sentences, dev_labels, _ = corpus_reader(validation_data)
    dev_dataset = NER_Dataset(tag2idx, dev_sentences, dev_labels, tokenizer_path=bert_tokenizer,
                              do_lower_case=do_lower_case)

    # print(len(train_dataset))
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return train_iter, eval_iter, tag2idx

def generate_test_data(config, tag2idx, bert_tokenizer="bert-base", do_lower_case=True):
    test_data = config.data_dir + config.test_data
    test_sentences, test_labels, _ = corpus_reader(test_data)
    test_dataset = NER_Dataset(tag2idx, test_sentences, test_labels, tokenizer_path=bert_tokenizer,
                               do_lower_case=do_lower_case)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter

def generate_use_data(config, bert_tokenizer="bert-base", do_lower_case=True):
    use_data = config.data_dir + config.use_data
    use_sentences = corpus_reader_use(use_data)
    token_ids, attn_mask, org_tok_map, original_token, token_len= [], [], [], [], []
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer, do_lower_case=do_lower_case)
    for id in range(len(use_sentences)):
        if id==29:
            print('等等')
        token_ids_temp, len_temp, orig_to_tok_map_temp, original_token_temp = get_use_data(sentences=use_sentences, idx=id, tokenizer=tokenizer, do_lower_case=do_lower_case)
        token_ids.append(token_ids_temp)
        org_tok_map.append(orig_to_tok_map_temp)
        original_token.append(original_token_temp)
        token_len.append(len_temp)
    batch = []
    for i in range(len(use_sentences)):
        temp = (token_ids[i], token_len[i], org_tok_map[i], original_token[i])
        batch.append(temp)
    tok_ids, attn_mask, org_tok_map, sents = pad_use(batch)
    tok_ids = torch.LongTensor(tok_ids)
    attn_mask = torch.LongTensor(attn_mask)
    batch_use = [tok_ids, attn_mask, org_tok_map, sents]
    return batch_use

def train(train_iter, eval_iter, tag2idx, config, bert_model="bert-base-uncased"):
    # print('#Tags: ', len(tag2idx))
    unique_labels = list(tag2idx.keys())
    model = Bert_CRF.from_pretrained(bert_model, num_labels=len(tag2idx))
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    num_epoch = config.epoch
    # 防止显存不足，在不改变梯度下降次数的前提下缩小batch_size
    gradient_acc_steps = 1
    t_total = len(train_iter) // gradient_acc_steps * num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    global_step = 0
    model.zero_grad()
    model.train()
    training_loss = []
    validation_loss = []
    train_iterator = trange(num_epoch, desc="Epoch", disable=0)
    start_time = timeit.default_timer()

    for epoch in (train_iterator):
        epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)
        tr_loss = 0.0
        tmp_loss = 0.0
        model.train()
        for step, batch in enumerate(epoch_iterator):
            s = timeit.default_timer()
            token_ids, attn_mask, _, labels, _, _ = batch
            # print(labels)
            inputs = {'input_ids': token_ids.to(device),
                      'attn_masks': attn_mask.to(device),
                      'labels': labels.to(device)
                      }
            loss = model(**inputs)
            loss.backward()
            tmp_loss += loss.item()
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if step == 0:
                print('\n%s Step: %d of %d Loss: %f' % (
                    str(datetime.datetime.now()), (step + 1), len(epoch_iterator), loss.item()))
            if (step + 1) % 100 == 0:
                print('%s Step: %d of %d Loss: %f' % (
                    str(datetime.datetime.now()), (step + 1), len(epoch_iterator), tmp_loss / 1000))
                tmp_loss = 0.0

        print("Training Loss: %f for epoch %d" % (tr_loss / len(train_iter), epoch))
        training_loss.append(tr_loss / len(train_iter))
        # '''
        # Y_pred = []
        # Y_true = []
        val_loss = 0.0
        model.eval()
        writer = open(config.apr_dir + 'prediction_' + str(epoch) + '.csv', 'w')
        for i, batch in enumerate(eval_iter):
            token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
            # attn_mask.dt
            inputs = {'input_ids': token_ids.to(device),
                      'attn_masks': attn_mask.to(device)
                      }

            dev_inputs = {'input_ids': token_ids.to(device),
                          'attn_masks': attn_mask.to(device),
                          'labels': labels.to(device)
                          }
            with torch.torch.no_grad():
                tag_seqs = model(**inputs)
                tmp_eval_loss = model(**dev_inputs)
            val_loss += tmp_eval_loss.item()
            # print(labels.numpy())
            y_true = list(labels.cpu().numpy())
            for i in range(len(sorted_idx)):
                o2m = org_tok_map[i]
                pos = sorted_idx.index(i)
                for j, orig_tok_idx in enumerate(o2m):
                    writer.write(original_token[i][j] + '\t')
                    writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                    pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                    if pred_tag == 'X':
                        pred_tag = 'O'
                    writer.write(pred_tag + '\n')
                writer.write('\n')

        validation_loss.append(val_loss / len(eval_iter))
        writer.flush()
        print('Epoch: ', epoch)
        command = "python conlleval.py < " + config.apr_dir + "prediction_" + str(epoch) + ".csv"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        result = process.communicate()[0].decode("utf-8")
        print(result)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss / len(train_iter),
        }, config.apr_dir + 'model_' + str(epoch) + '.pt')

    total_time = timeit.default_timer() - start_time
    print('Total training time: ', total_time)
    return training_loss, validation_loss

'''
    raw_text should pad data in raw data prediction
'''

def test(config, test_iter, model, unique_labels, test_output):
    model.eval()
    csvFile = open(config.apr_dir + test_output, "w", newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    #writer = open(config.apr_dir + test_output, 'w', encoding='utf-8')
    for i, batch in enumerate(test_iter):
        token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
        # attn_mask.dt
        inputs = {'input_ids': token_ids.to(device),
                  'attn_masks': attn_mask.to(device)
                  }
        with torch.torch.no_grad():
            tag_seqs = model(**inputs)
        y_true = list(labels.cpu().numpy())
        for i in range(len(sorted_idx)):
            o2m = org_tok_map[i]
            pos = sorted_idx.index(i)
            for j, orig_tok_idx in enumerate(o2m):
                #writer.write(original_token[i][j] + '\t')
                #writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                if pred_tag == 'X':
                    pred_tag = 'O'
                #writer.write(pred_tag + '\n')
                temp = []
                #if original_token[i][j] == 'confirmed':
                    #print('123')
                temp.append(original_token[i][j])
                temp.append(unique_labels[y_true[pos][orig_tok_idx]])
                temp.append(pred_tag)
                writer.writerow(temp)
            writer.writerow('\n')
    csvFile.close()
    command = "python conlleval.py < " + config.apr_dir + test_output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    result = process.communicate()[0].decode("utf-8")
    print(result)

def use(config, model, batch_use, unique_labels):
    # batch_use = [tok_ids, attn_mask, org_tok_map, sents]
    model.eval()
    writer = open(config.apr_dir + config.use_out, 'w', encoding='utf-8')
    token_ids = batch_use[0]
    attn_mask = batch_use[1]
    org_tok_map = batch_use[2]
    original_token = batch_use[3]
    inputs = {'input_ids': token_ids.to(device),
              'attn_masks': attn_mask.to(device)
              }
    with torch.torch.no_grad():
        tag_seqs = model(**inputs)
    for i in range(len(token_ids)):
        o2m = org_tok_map[i]
        for j, orig_tok_idx in enumerate(o2m):
            writer.write(original_token[i][j])
            pred_tag = unique_labels[tag_seqs[i][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            writer.write(pred_tag + '\n')
        writer.write('\n')
    writer.flush()
    print("预测完毕")

def parse_raw_data(padded_raw_data, model, unique_labels, out_file_name='raw_prediction.csv'):
    model.eval()
    token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = padded_raw_data
    # attn_mask.dt
    writer = open(out_file_name, 'w')
    inputs = {'input_ids': token_ids.to(device),
              'attn_masks': attn_mask.to(device)
              }
    with torch.torch.no_grad():
        tag_seqs = model(**inputs)
    y_true = list(labels.cpu().numpy())
    for i in range(len(sorted_idx)):
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            writer.write(original_token[i][j] + '\t')
            writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            writer.write(pred_tag + '\n')
        writer.write('\n')
    print("Raw data prediction done!")

def show_graph(training_loss, validation_loss, resource_dir):
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    plt.plot(range(1, len(training_loss) + 1), validation_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss Vs Testing Loss")
    plt.legend()
    plt.show()
    plt.savefig(resource_dir + 'Loss.png')

def load_model(config, do_lower_case=True):
    f = open(config.apr_dir + 'tag2idx.pkl', 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())
    model = Bert_CRF.from_pretrained(config.bert_model, num_labels=len(tag2idx))
    checkpoint = torch.load(config.apr_dir + config.model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    global bert_tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=do_lower_case)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model, bert_tokenizer, unique_labels, tag2idx

def usage(parameter):
    parameter.print_help()

    print("Example usage (training):\n", \
          "\t python bert_crf.py --mode train ")

    print("Example usage (testing):\n", \
          "\t python bert_crf.py --mode test ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
log_soft = F.log_softmax
seed_torch()
if __name__ == "__main__":
    user_input = OptionParser()
    user_input.add_option("--mode", dest="model_mode", metavar="string", default='test',
                          help="mode of the model (required)")
    (options, args) = user_input.parse_args()
    if options.model_mode == "train":         
        train_iter, eval_iter, tag2idx = generate_training_data(config=config, bert_tokenizer=config.bert_model,
                                                                do_lower_case=True)
        t_loss, v_loss = train(train_iter, eval_iter, tag2idx, config=config, bert_model=config.bert_model)
        show_graph(t_loss, v_loss, config.apr_dir)
    elif options.model_mode == "test":
        model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
        test_iter = generate_test_data(config, tag2idx, bert_tokenizer=config.bert_model, do_lower_case=True)
        print('test len: ', len(test_iter))
        test(config, test_iter, model, unique_labels, config.test_out)
    elif options.model_mode == "use":
        model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
        batch_use = generate_use_data(config, bert_tokenizer=config.bert_model, do_lower_case=True)
        use(config, model, batch_use, unique_labels)
    else:
        usage(user_input)
