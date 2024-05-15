#实验：统计数据集的相关信息
import os

# 统计activity condition signal三种实体以及token的总量
def statistic_entity():
    data_dir = '../data/'
    file_name = os.listdir(data_dir)
    file_name.remove('tag_2.txt')
    activity_num = 0
    activity_I_num = 0
    condition_num = 0
    condition_I_num = 0
    signal_num = 0
    signal_I_num = 0
    token_num = 0
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
                activity_num += 1
            elif pair[1] == 'B-signal':
                signal_num += 1
            elif pair[1] == 'B-condition':
                condition_num += 1
            elif pair[1] == 'I-activity':
                activity_I_num += 1
            elif pair[1] == 'I-condition':
                condition_I_num += 1
            elif pair[1] == 'I-signal':
                signal_I_num += 1
            if pair[0] not in ['.', ',', '?', '!', ';', ':']:
                token_num += 1


# 具体到单个文件，统计单个文件中的token数量，以及平均数 最大值 最小值
def statistic_token_per_file():
    data_dir = '../data/'
    file_name = os.listdir(data_dir)
    file_name.remove('tag_2.txt')
    file_name_and_token_num = []
    for cur_file_name in file_name:
        cur_content = []
        cur_token_num = 0
        cur_file = data_dir + cur_file_name
        with open(cur_file, encoding='utf-8') as file:
            content = file.read()
            content = content.split('\n')
            for line in content:
                cur_content.append(line.split())
            for pair in cur_content:
                if pair != [] and pair[0] not in ['.', ',', '?', '!', ';', ':']:
                    cur_token_num += 1
            file_name_and_token_num.append([cur_file_name, cur_token_num])

def activity_temporal_relationship_statistic():
    class_0 = 0 #无关系
    class_1 = 0 #A先B后
    class_2 = 0 #B先A后
    class_3 = 0 #选择
    class_4 = 0 #并发
    file_name = '../data/tag_2.txt'
    file_name_and_relation_seq = []
    #只统计部分文件的时序关系数量
    file_choice = [10, 21, 30, 41, 52, 63, 65, 66, 67, 0, 1, 2]
    for i in range(0, len(file_choice)):
        file_choice[i] = str(file_choice[i])+'_data1.txt'
    with open(file_name, encoding='utf-8') as file:
        content = file.read()
        content = content.split('\n')
    content = content[0:-1]
    for line in content:
        temp = line.split()
        if(temp[0] in file_choice):
            file_name_and_relation_seq.append(temp)
    file_name_and_relation_seq = file_name_and_relation_seq[:-1]
    for file, seq in file_name_and_relation_seq:
        for i in seq:
            if i == '1':
                class_1 += 1
            elif i == '2':
                class_2 += 1
            elif i == '3':
                class_3 += 1
            elif i == '4':
                class_4 += 1
            elif i == '0':
                class_0 += 1

    print(1)

if __name__ == '__main__':
    activity_temporal_relationship_statistic()


