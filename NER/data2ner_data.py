# -*- coding: utf-8 -*-
'''
把将要活动体识别的数据转化为一个token占一行的格式
'''
import os

data_dir = 'data/李文鑫data/Texts/'
output_dir = 'data/李文鑫data/data1/无标签/'
data_files = os.listdir(data_dir)
punctuation = '!,.?;:'
for data_file in data_files:
    result_list = []
    with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            temp1 = line.split()
            temp2 = []
            for i in range(len(temp1)):
                if temp1[i][-1] in punctuation:
                    temp2.append(temp1[i][:-1])
                    temp2.append(temp1[i][-1])
                else:
                    temp2.append(temp1[i])
                i += 1
            result_list += temp2
    output_data_name = data_file[:-4] + '_无标签' + '.txt'
    with open(output_dir + output_data_name, "w", encoding='utf-8') as f:
        for i in result_list:
            f.write(i)
            f.write('\n')




