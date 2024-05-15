#实验：统计存在时序关系的活动之间的距离，证明活动时序关系具有局部性特征

data_dir = '../data/tag_2.txt'
data = []
distance = []
distance_statistic = {}
with open(data_dir, encoding='utf-8') as file:
    content = file.read()
content = content.split('\n')
for line in content:
    data.append(line.split())
data = data[:-1]
for file in data:
    seq_len = len(file[1])
    for i in range(1, 50):
        if int(i * (i-1)/2) == seq_len:
            activity_num = i
            break
    for i in range(0, seq_len):
        if file[1][i] in ['1', '2', '3', '4']:
            lun = activity_num - 1
            number = i + 1
            while number > 0:
                number = number - lun
                lun -= 1
            number += lun + 1
            distance.append(number)
for cur_dis in distance:
    if cur_dis in distance_statistic:
        distance_statistic[cur_dis] += 1
    else:
        distance_statistic[cur_dis] = 1
print(1)



