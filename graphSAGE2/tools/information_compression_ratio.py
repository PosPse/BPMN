# 实验：信息压缩率计算：压缩后大小/压缩前大小，数字越小说明压缩信息越多
import os

data_dir = '../data/'
file_name = os.listdir(data_dir)
file_name.remove('tag_2.txt')
pre_compression = 0
after_compression = 0
total_content = []
for cur_file_name in file_name:
    cur_file = data_dir + cur_file_name
    with open(cur_file, encoding='utf-8') as file:
        content = file.read()
        content = content.split('\n')
        for line in content:
            total_content.append(line.split())

for pair in total_content:
    if pair != []:
        if pair[1] == 'B-activity':
            after_compression += 1
        elif pair[1] == 'B-signal':
            after_compression += 1
        elif pair[1] == 'B-condition':
            after_compression += 1
        elif pair[1] == 'I-activity':
            pass
        elif pair[1] == 'I-condition':
            pass
        elif pair[1] == 'I-signal':
            after_compression += 1
        elif pair[1] == 'punctuation':
            pass
        elif pair[1] == 'O':
            pass
        else:
            print("未知的tag:"+pair[1]+'所属文件:'+pair[0])
        pre_compression += 1



