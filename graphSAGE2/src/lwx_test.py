import torch
import re
import os
'''
# 打印所有活动，给活动标号
require = input('输入查询:')
while require != 'stop':
    require = input('输入查询:')
    require += '_data1.txt'
    file = open('../data/'+require, 'r', encoding='utf-8')
    content = file.readlines()
    temp = ''
    count = 1
    for line in content:
        line = re.sub('\n', '', line)
        line = line.split(" ")
        if line[1] == 'B-activity':
            temp = str(count)+': '
            temp += line[0]+ ' '
        elif line[1] == 'I-activity':
            temp = temp + line[0]
            temp = temp + ' '
        else:
            if temp != '':
                print(temp)
                count += 1
                temp = ''
'''
# 检查多个document之间是否有潜在的重复文件
total_content = {}
data_files = os.listdir('../data/')
for current in data_files:
    current_dict = {}
    current_content = []
    file = open('../data/' + current, 'r', encoding='utf-8')
    content = file.readlines()
    for line in content:
        line = re.sub('\n', '', line)
        line = line.split(" ")
        current_content.append(line[0])
    total_content[current] = current_content

total = list(total_content.keys())
for item in total:
    potential = list(total_content.keys())
    print("开始检查:"+item+'{', end='')
    for i in range(0, 10):
        if i==0:
            potential.remove(item)
        for content in potential:
            if total_content[item][i] not in total_content[content]:
                potential.remove(content)
    print(potential, end='')
    print('}')


