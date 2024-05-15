'''
从文件夹中读取数据文件，并将文件处理成信号词+标点+'[activity]'+'[condition]'的形式
输出文件中对应的活动体以及活动体序号
'''
import os
import re

import torch
# from graphviz import Digraph

def get_data(data_dir):
    data_files = os.listdir(data_dir)
    file_list = []
    result_list = []
    tag = ['I-activity', 'B-activity', 'B-signal', 'I-signal', 'O', 'punctuation', 'B-condition', 'I-condition']
    for data_file in data_files:
        data_list = []
        data_list.append(data_file)
        with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = re.sub('\n', '', line)
                temp = line.split(" ")
                # 检验数据标注的正确性
                if temp[1] not in tag:
                    print(data_file + ':'+temp[0] + temp[1])
                data_list.append(temp)
        file_list.append(data_list)

    for i in range(0,len(file_list)):
        temp = file_list[i]
        para_temp = []
        for couple in temp:
            if couple[1] == 'B-activity':
                para_temp.append('[activity]')
            elif couple[1] == 'I-activity':
                pass
            elif couple[1] in ['B-signal', 'I-signal']:
                para_temp.append(couple[0])
            elif couple[1] in ['O']:
                pass
            elif couple[1] == 'B-condition':
                para_temp.append('[condition]')
            elif couple[1] == 'I-condition':
                pass
            elif couple[1] in ['punctuation']:
                pass
                # para_temp.append(couple[0])
            else:
                para_temp.append(couple)
        result_list.append(para_temp)
        tag_dir = 'data/tag/tag.txt'
        tag_list = []
        with open(tag_dir, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.split(" ")
                file_name = line[0]
                tag = line[1]
                tag_list.append([file_name, tag])
    return result_list, tag_list

def get_activity_order():
    data_dir = 'data\\'
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        activities = []
        activity = []
        with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
            print("开始处理:"+data_file)
            for line in reader:
                line = re.sub('\n', '', line)
                temp = line.split(" ")
                if temp[1] == 'B-activity':
                    if len(activity) != 0:
                        activities.append(activity)
                        activity = []
                    activity.append(temp[0])
                if temp[1] == 'I-activity':
                    activity.append(temp[0])
                if temp[1] != 'I-activity' and temp[1] != 'B-activity':
                    if len(activity) != 0:
                        activities.append(activity)
                        activity = []
        count = 1
        for act in activities:
            print(str(count) + ':', end='')
            for word in act:
                print(word+' ', end='')
            print('\n')
            count += 1
        print("当前文件结束\n")
        next_file = input()

def find_file_name():
    data_dir = 'data\\'
    data_files = os.listdir(data_dir)
    total = []
    for data_file in data_files:
        current = []
        content = ''
        with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
            current.append([data_file])
            for line in reader:
                line = re.sub('\n', '', line)
                temp = line.split(" ")
                content += (temp[0] + ' ')
        current.append([content])
        total.append(current)
    while True:
        is_find = False
        key = input("输入查询片段:")
        for i in range(0, len(total)):
            if key in total[i][1][0]:
                print('找到，在'+total[i][0][0])
                is_find = True
                break
        if is_find is False:
            print("找不到")

def write_tag():
    while True:
        file_id = input('输入文件序号')
        if file_id == 'end':
            break
        file_name = file_id + '_data1.txt'
        activity_num = int(input('输入活动体数量'))
        tag_seq = [0]*int((activity_num * (activity_num - 1)/2))
        while True:
            temp = input('输入活动信息:end结束')
            if temp == 'end':
                break
            temp = temp.split(" ")
            pre_activity_id = int(temp[0])
            latter_activity_id = int(temp[1])
            ralation = int(temp[2])
            index = 0
            if pre_activity_id - 1 >= 1:
                for i in range(activity_num - pre_activity_id + 1 , activity_num):
                    index += i
            index += (latter_activity_id - pre_activity_id - 1)
            tag_seq[index] = ralation
        #draw(file_id, tag_seq, activity_num)
        tag_dir = 'data\\tag_2.txt'
        tag_file = open(tag_dir, 'a')
        temp = (file_name + ' ' + re.sub(' ','',re.sub(',', '', re.sub( ']', '', re.sub('\[', '', str(tag_seq))))))
        temp += '\n'
        tag_file.write(temp)
        tag_file.close()

# def draw(name, tag_seq, activity_num):
#     g = Digraph(name=name, format="png")
#     for i in range(1, activity_num+1):
#         g.node(name=str(i), label=str(i), shape='circle')
#     tag_seq_list = []
#     for i in range(0, len(tag_seq)):
#         tag_seq_list.append(tag_seq[i])
#     for i in range(0, len(tag_seq_list)):
#         index = i + 1
#         former_activity_id = 0
#         latter_activity_id = 0
#         if tag_seq_list[i] in [1, 2]:
#             grad = activity_num-1
#             former = index
#             while index>0:
#                 former = index
#                 former_activity_id += 1
#                 index -= grad
#                 grad -= 1
#             latter_activity_id = former_activity_id + former
#             if tag_seq_list[i] == 1:
#                 g.edge(str(former_activity_id), str(latter_activity_id), color="#272727")
#             if tag_seq_list[i] == 2:
#                 g.edge(str(latter_activity_id), str(former_activity_id), color="#272727")
#     g.render(filename=name, directory="figure", view=False)

def token2id(tokens, file_name):
    vocab_list = []
    result = []
    # 先变小写
    for i in range(0, len(tokens)):
        if tokens[i] != '[PAD]':
            tokens[i] = tokens[i].casefold()
    with open('vocab.txt', 'r', encoding='utf-8') as reader:
        for line in reader:
            line = re.sub('\n', '', line)
            vocab_list.append(line)
    for token in tokens:
        if token in vocab_list:
            result.append(vocab_list.index(token))
        else:
            print(token+"不在词表内,文件名:"+file_name)
            result.append(vocab_list.index('[UNK]'))
    return result

def data2padding(tokens, max_len):
    if len(tokens) > max_len:
        print('已经超出最大长度')
    else:
        pad_num = max_len - len(tokens)
        for i in range(0,pad_num):
            tokens.append('[PAD]')
    return tokens

def tag2padding(tag, max_len):
    max_len_tag = max_len
    padding_num = max_len_tag - len(tag)
    for i in range(0, padding_num):
        tag += 'p'
    return tag

def data_concat_padding(data, max_len):
    padding_num = int(max_len - data.shape[1])
    padding_content = torch.Tensor([0]*data.shape[2]).unsqueeze(0).unsqueeze(0)
    padding_content = padding_content.repeat(1, padding_num, 1)
    if torch.cuda.is_available():
        padding_content = padding_content.cuda()
    data = torch.concat((data, padding_content), 1)
    return data

def assess(x, label):
    class_output = []
    label_output = []
    for i in range(0, len(x)):
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
    for i in range(0, len(label)):
        max_index = 0
        if label[i][1] >= label[i][max_index]:
            max_index = 1
        if label[i][2] >= label[i][max_index]:
            max_index = 2
        if max_index == 0:
            label_output.append(0)
        if max_index == 1:
            label_output.append(1)
        if max_index == 2:
            label_output.append(2)

    if len(class_output)!=len(label_output):
        print('输出和标签数量不一致！')
    right = 0.
    error = 0.
    for i in range(0, len(label_output)):
        if class_output[i] == label_output[i]:
            right += 1.
        else:
            error += 1.
    accuracy = right/(right+error)
    print('综合正确率：'+str(accuracy))

    total = 0.
    right = 0.
    for i in range(0, len(label_output)):
        if label_output[i] == 1 or label_output[i] == 2:
            total += 1.
            if class_output[i] == label_output[i]:
                right += 1.
    recall = right/total
    print('关键样本召回率：' + str(recall))

    error = 0.
    right = 0.
    for i in range(0, len(label_output)):
        if label_output[i] == 1 or label_output[i] == 2:
            if class_output[i] == label_output[i]:
                right += 1.
            else:
                error += 1.
    accuracy_2 = right / (right+error)
    print('关键样本正确率：' + str(accuracy_2))

def get_tag_rate(data_dir):
    tag_list = []
    num_1 = 0.
    num_2 = 0.
    num_3 = 0.
    with open(data_dir, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.split(" ")
            tag_list.append(line[1])
    for i in tag_list:
        for j in i:
            if j == '0':
                num_1 += 1.
            elif j == '1':
                num_2 += 1.
            elif j == '2':
                num_3 += 1.
    rate_1 = 1. /(num_1/(num_1 + num_2 + num_3))
    rate_2 = 1. /(num_2/(num_1 + num_2 + num_3))
    rate_3 = 1. /(num_3/(num_1 + num_2 + num_3))
    print('1')

if __name__ == "__main__":
    write_tag()
















