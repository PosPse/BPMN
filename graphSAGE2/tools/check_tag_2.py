'''
活动关系的合规性检查
对比发现部分活动关系的标签序列长度不符合原数据的实际情况
'''

import os
import re

data_dir = '../data/'
tag_2_file = '../data/tag_2.txt'
data_files = os.listdir(data_dir)
file_list = []
tag_list = []
for data_file in data_files:
    data_list = []
    if data_file != 'tag_2.txt':
        data_list.append(data_file)
        with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = re.sub('\n', '', line)
                temp = line.split(" ")
                data_list.append(temp)
        file_list.append(data_list)

with open(tag_2_file, 'r', encoding='utf-8') as reader:
    for line in reader:
        line = re.sub('\n', '', line)
        temp = line.split(" ")
        tag_list.append(temp)

# 开始验证
for file in file_list:
    activity_num = 0
    file_name = file[0]
    if file_name == '69_data1.txt':
        print(1)
    for i in range(1, len(file)):
        token, entity_tag = file[i]
        if entity_tag == 'B-activity':
            activity_num += 1
    tag_len = activity_num/2*(activity_num-1)
    for tag_2_file_name, tag_seq in tag_list:
        if tag_2_file_name == file_name:
            if len(tag_seq) != tag_len:
                print(file_name+'错误')


