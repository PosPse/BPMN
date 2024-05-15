# -*- coding: utf-8 -*-
'''
自动化给一部分数据加上标签
即转换为'/data/李文鑫data/有标签'下的数据
'''
from stanfordcorenlp import StanfordCoreNLP
import os
import re

def get_words_from_sentence_parse(sentence):
    left = right = 0
    result = []
    while True:
        while sentence[left] != '(':
            left += 1
            if left == len(sentence):
                return  result
        right = left
        while sentence[right] != ')':
            if sentence[right] == '(':
                left = right
            right += 1
            if right == len(sentence):
                return  result
        result.append(sentence[left+1: right])
        left = right

def get_activity_index_from_sentence(sentence, line, target='VP' ):
    sentence = sentence.replace('\n', '').replace('\r', '')
    NPs = re.finditer(target, sentence)
    NP_index = []
    result = []
    index_pair = []
    for NP in NPs:
        left = 1
        start = NP.end() - 3
        current = start + 3
        while left != 0 and current != len(sentence)-1:
            current += 1
            if sentence[current]=='(':
                left += 1
            elif sentence[current]==')':
                left -= 1
        NP_index.append((start, current))
    for index in NP_index:
        temp = sentence[index[0]: index[1]+1]
        temp = get_words_from_sentence_parse(temp)
        for i in temp:
            i = nlp.word_tokenize(i)
            result.append(i[1])
    result_index = line_index = 0
    left = right = 0
    while True:
        if result_index >= len(result) or line_index >= len(line):
            return index_pair
        if result[result_index] == line[line_index]:
            left = right = line_index
            while result_index < len(result) and line_index < len(line) and result[result_index] == line[line_index]:
                result_index += 1
                line_index += 1
                is_append = True
            right = line_index - 1
            if is_append:
                index_pair.append((left, right))
                is_append = False
        line_index += 1
        if result_index >= len(result) or line_index >= len(line):
            return index_pair

if __name__ == "__main__":
    data_dir = 'data/李文鑫data/test/'
    output_dir = 'data/李文鑫data/data1/有标签/'
    data_files = os.listdir(data_dir)
    print('共有'+str(len(data_files))+'个文件需要处理')
    nlp = StanfordCoreNLP(r'G:\stanford-corenlp-4.3.2',lang='en')
    num = 1
    for data_file in data_files:
        print("开始处理，文件"+str(num))
        output_data_name = data_file[:-4] + '_data1' + '.txt'
        with open(data_dir + data_file, 'r', encoding='utf-8') as reader:
            sentence_list = []
            result = []
            for line in reader:
                if line != '':
                    line = line.replace('\n', '')
                    sentence = line
                    line = nlp.word_tokenize(line)
                    result_temp = [line, ['O']*len(line)]
                    for token in line:
                        sentence_list.append(token)
                    sentence_parse = nlp.parse(sentence)
                    activity_index = get_activity_index_from_sentence(sentence = sentence_parse, line = line, target='VP')
                    for index in activity_index:
                        result_temp[1][index[0]] = 'B-activity'
                        for i in range(index[0]+1, index[1]+1):
                            result_temp[1][i] = 'I-activity'
                    result.append((result_temp[0],result_temp[1]))
        with open(output_dir + output_data_name, "w", encoding='utf-8') as f:
            print("开始写入:")
            for i in result:
                for j in range(0,len(i[0])):
                    f.write(i[0][j])
                    f.write(' ')
                    f.write(i[1][j])
                    f.write('\n')
            print("写入完毕:")
        num += 1
    nlp.close()
