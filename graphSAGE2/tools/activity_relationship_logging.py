#录入活动关系的数据
data_dir = '../data/tag_2.txt'
f = open(data_dir,"a",encoding = "utf - 8")
while True:
    document_name = input("输入文件的编号(0退出):")
    if document_name == '0':
        break
    document_name = document_name + '_data1.txt'
    activity_num = int(input("输入活动的数量:"))
    tag_seq = '0' * int(activity_num * (activity_num - 1) / 2)
    tag_seq = list(tag_seq)
    while True:
        cur_activity_relation = input("输入活动关系:activity1 activity2 relation(0 0 0弹出)").split()
        first_activity = int(cur_activity_relation[0])
        last_activity = int(cur_activity_relation[1])
        relation = cur_activity_relation[2]
        if first_activity == 0:
            break
        cur_lun = activity_num - 1
        cur_index = 0
        for i in range(0, first_activity - 1):
            cur_index += cur_lun
            cur_lun -= 1
        cur_index += last_activity - first_activity
        cur_index -= 1
        tag_seq[cur_index] = relation
    f.write(document_name + ' ')
    for i in tag_seq:
        f.write(i)
    f.write('\n')
f.close()


