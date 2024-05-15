# 将分布在多个文件上的数据整理到一个文件中，分为训练集、验证集和测试集

import os
import re
import random
import csv

data_dir = 'data\\李文鑫data\\有标签\\'
data_files = os.listdir(data_dir)
print("待处理文件数:"+str(len(data_files)))
train_name = 'train.csv'
dev_name = 'dev.csv'
prediction_name = 'prediction.csv'
temp_list = []
total_list = [[]]
train_list = []
dev_list = []
prediction_list = []
current_num = 1
total_token = 0

O_num = 0
activity_num = 0
condition_num = 0
signal_num = 0
punctuation_num = 0
for data_file in data_files:
    with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
        print("开始处理:"+str(current_num))
        for line in reader:
            line = re.sub('\n', '', line)
            temp = line.split(" ")
            if temp[1] == 'O': O_num += 1
            elif temp[1] == 'punctuation': punctuation_num += 1
            elif temp[1] in ['B-activity', 'I-activity']: activity_num += 1
            elif temp[1] in ['B-condition', 'I-condition']: condition_num += 1
            elif temp[1] in ['B-signal', 'I-signal']: signal_num += 1
            else: print('未识别的标签：'+temp[1])
            temp_list.append(temp)
        current_num += 1
total_token = O_num + activity_num + condition_num + signal_num + punctuation_num
data_num = 0
for temp in temp_list:
    if temp[0] == '.' or temp[0] == '?' or temp[0] == '!':
        total_list[data_num].append(temp)
        data_num += 1
        total_list.append([])
    else:
        total_list[data_num].append(temp)
total_num = len(total_list)
print("数据数量共有："+ str(total_num))
train_num = int(total_num * 0.8)
dev_num = int(total_num * 0.1)
prediction_num = total_num - train_num - dev_num
print("训练集数量共有："+ str(train_num))
print("验证集数量共有："+ str(dev_num))
print("测试集数量共有："+ str(prediction_num))

train_index = []
train_index.append(0)
train_index.append(train_num)
dev_index = []
dev_index.append(train_index[1])
dev_index.append(train_index[1] + dev_num)
prediction_index = []
prediction_index.append(dev_index[1])
prediction_index.append(total_num)
random.shuffle(total_list)
train_list = total_list[train_index[0]: train_index[1]]
dev_list = total_list[dev_index[0]: dev_index[1]]
prediction_list = total_list[prediction_index[0]: prediction_index[1]]

f = open(train_name,'w',encoding='utf-8', newline='')
csv_writer = csv.writer(f)
for data in train_list:
    for temp in data:
        csv_writer.writerow(temp)
    csv_writer.writerow('')
f.close()
f = open(dev_name,'w',encoding='utf-8', newline='')
csv_writer = csv.writer(f)
for data in dev_list:
    for temp in data:
        csv_writer.writerow(temp)
    csv_writer.writerow('')
f.close()
f = open(prediction_name,'w',encoding='utf-8', newline='')
csv_writer = csv.writer(f)
for data in prediction_list:
    for temp in data:
        csv_writer.writerow(temp)
    csv_writer.writerow('')
f.close()
'''
num_B_activity = 0
num_I_activity = 0
num_B_condition = 0
num_I_condition = 0
num_B_signal = 0
num_I_signal = 0
for sentence in prediction_list:
    for token in sentence:
        if token[1] == 'B-activity':
            num_B_activity += 1
        if token[1] == 'I-activity':
            num_I_activity += 1
        if token[1] == 'B-condition':
            num_B_condition += 1
        if token[1] == 'I-condition':
            num_I_condition += 1
        if token[1] == 'B-signal':
            num_B_signal += 1
        if token[1] == 'I-signal':
            num_I_signal += 1
print(1)
'''

