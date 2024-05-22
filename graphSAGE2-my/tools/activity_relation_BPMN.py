import copy
from collections import deque
#pm4py存放图片的默认路径在此：C:\Users\Administrator\AppData\Local\Temp
#从活动时序关系到BPMN图
import os

import pm4py
from graphviz import Digraph


#先把一个标签序列中的选择和并发关系找出来，先解决选择和并发
def find_34_in_seq(activity_num, tag_seq):
    class_3 = []
    class_4 = []

    for former_index in range(1, activity_num):
        class_3.append([former_index])
        class_4.append([former_index])
        for later_index in range(former_index + 1, activity_num + 1):
            lun = activity_num - 1
            seq_index = 0
            for i in range(1, former_index):
                seq_index += lun
                lun -= 1
            seq_index += later_index - former_index - 1
            if(tag_seq[seq_index] == '3'):#选择
                class_3[-1].append(later_index)
            elif (tag_seq[seq_index] == '4'): #并发
                class_4[-1].append(later_index)

    #删除非必要元素
    for i in range(0, activity_num - 1):
        if len(class_3[i]) > 1:
            for j in range(1, len(class_3[i])):
                if class_3[i][j] != activity_num:
                    class_3[class_3[i][j] - 1] = [class_3[i][j]]
        if len(class_4[i]) > 1:
            for j in range(1, len(class_4[i])):
                if class_4[i][j] != activity_num:
                    class_4[class_4[i][j] - 1] = [class_4[i][j]]

    return class_3, class_4

# 寻找哪些结点和begin相连
def find_begin_and_end_to_node(activity_num, tag_seq):
    begin_node_list = []
    end_node_list = []
    for i in range(1, activity_num+1):
        begin_node_list.append(i)
        end_node_list.append(i)

    for former_index in range(1, activity_num):
        for later_index in range(former_index + 1, activity_num + 1):
            lun = activity_num - 1
            seq_index = 0
            for i in range(1, former_index):
                seq_index += lun
                lun -= 1
            seq_index += later_index - former_index - 1
            if(tag_seq[seq_index] == '1'):#选择
                if former_index in end_node_list:
                    end_node_list.remove(former_index)
                if later_index in begin_node_list:
                    begin_node_list.remove(later_index)
            elif (tag_seq[seq_index] == '2'): #并发
                if former_index in begin_node_list:
                    begin_node_list.remove(former_index)
                if later_index in end_node_list:
                    end_node_list.remove(later_index)

    return begin_node_list, end_node_list

# 给序列求activity_num
def get_activity_num(tag_seq):
    tag_seq_len = len(tag_seq)
    for i in range(2, 100):
        if int((i * (i-1))/2) == tag_seq_len:
            return i

# 输入边表和选择/并发结点群组,返回收敛点id,再返回连到收敛点的结点组,要传int进来
def find_convergency_node(name, relation_tag_matrix, node_set):
    print("当前处理文件序号:" + name)
    print("处理结点组列表:")
    for node in node_set:
        print(node)
    if name == '15':
        print(1)
    # 冻结node_set,执行完全部节点
    finished_set = set()
    undo = 1
    while undo == 1:
        undo = 0
        for i in range(0, len(relation_tag_matrix)):
            if i in node_set or i in finished_set:
                continue
            is_prepared = True
            for j in range(0, len(relation_tag_matrix[0])):
                if relation_tag_matrix[i][j] == 2:
                    is_prepared = False
                    break
            if is_prepared:
                undo = 1
                finished_set.add(i)
                for j in range(0, len(relation_tag_matrix[0])):
                    if relation_tag_matrix[i][j] == 1:
                        relation_tag_matrix[i][j] = 0
                        relation_tag_matrix[j][i] = 0

    copy_relation_tag_matrix = copy.deepcopy(relation_tag_matrix)
    while len(node_set) > 1:
        # 统计当前可执行结点的集合
        current_do_set = set()
        for node in node_set:
            isPrepared = True
            for i in range(0, len(relation_tag_matrix)):
                if relation_tag_matrix[node][i] == 2:
                    isPrepared = False
                    break
            if isPrepared:
                current_do_set.add(node)
        # 把它们执行掉，然后把它们接下来要执行的结点弄进来
        for node in current_do_set:
            for i in range(0, len(relation_tag_matrix)):
                if relation_tag_matrix[node][i] == 1:
                    node_set.add(i)
                    relation_tag_matrix[node][i] = 0
                    relation_tag_matrix[i][node] = 0
            node_set.remove(node)
        # 看一看是不是收敛到一个结点上了
        if len(node_set) == 1:
            for node in node_set:
                before_convergency_nodes = []
                for i in range(0, len(copy_relation_tag_matrix[0])):
                    if copy_relation_tag_matrix[node][i] == 2:
                        before_convergency_nodes.append(i)
                return node, before_convergency_nodes
    print("代码写错球了")

#从活动时序关系到BPMN图
def activity_relationship_to_BPMN(name, activity_num, tag_seq, activity_list):
    g = Digraph(name=name, format="png")
    # print("name:"+name)
    # if name == "24":
    #     print(1)
    node_id = 1
    # 把所有活动的node画出来，node_id为activity_list的索引
    for i in range(1, activity_num + 1):
        g.node(name=str(node_id), label=activity_list[node_id - 1], shape='polygon')# 节点的name为活动的id
        node_id += 1
    # 绘制开始和结束的node，并记录node的id,开始node的id是0，结束node的id是activity_num+1
    g.node(name='0', label='', shape='circle', color='#00FF00', style='filled')
    g.node(name=str(activity_num+1), label='', shape='circle', color='#FF8C00', style='filled')

    # 建立node代理机制，node有前代理和后代理两种代理模式，结点A指向结点B，本质上是结点A的后代理指向结点B的前代理，代理结点可能是结点本身。
    node_agency_former = {}
    node_agency_latter = {}
    # 谁连后代理是一个复杂的问题，后来者居上
    node_to_agency = {}
    # 某些并发/选择群组已经处理过，那么保存它们的前代理结点即可
    nodes_and_before_agency = {}
    # 代理初始化为结点本身
    for i in range(0, activity_num+2):
        node_agency_former[i] = str(i)
        node_agency_latter[i] = str(i)
        node_to_agency[i] = "-1"
    # 为了方便运算，把tag序列解析为一个矩阵
    relation_tag_matrix = []
    for i in range(0, activity_num+2):# 第0行是开始节点，最后一行是结束节点
        relation_tag_matrix.append([])
        for j in range(0, activity_num+2):
            relation_tag_matrix[-1].append(-1)
    # 开始解析tag序列
    index = 0
    for i in range(1, activity_num+1):
        for j in range(i + 1, activity_num+1):
            cur_relation = int(tag_seq[index])
            relation_tag_matrix[i][j] = cur_relation
            if (cur_relation == 1):
                relation_tag_matrix[j][i] = 2
            elif (cur_relation == 2):
                relation_tag_matrix[j][i] = 1
            else:
                relation_tag_matrix[j][i] = cur_relation
            index += 1

    begin_node_list, end_node_list = find_begin_and_end_to_node(activity_num, tag_seq)
    for node in begin_node_list:
        relation_tag_matrix[0][node] = 1
        relation_tag_matrix[node][0] = 2
    for node in end_node_list:
        relation_tag_matrix[node][activity_num+1] = 1
        relation_tag_matrix[activity_num+1][node] = 2

    draw_map = copy.deepcopy(relation_tag_matrix)
    is_finished = set()
    to_do_node = deque()
    to_do_node.append(0)
    before_proxy_num = 1
    after_proxy_num = 1
    while len(to_do_node) > 0:
        current_node_id = to_do_node.popleft()
        current_level_nodes = set()
        while current_node_id in is_finished:
            if len(to_do_node) > 0 :
                current_node_id = to_do_node.popleft()
            else:
                break
        if current_node_id in is_finished:
            break
        # 选择/并发群组字符串序列
        nodes_group = ""
        for i in range(1, activity_num + 2):
            if relation_tag_matrix[current_node_id][i] == 1 :
                current_level_nodes.add(i)
                nodes_group += str(i)
        if len(current_level_nodes) > 1 :
            # 如果在群组中的话，不用处理了，直接连接前代理
            if nodes_group in nodes_and_before_agency:
                g.edge(str(current_node_id), nodes_and_before_agency[nodes_group], color="#272727")
                for i in range(0, activity_num + 2):
                    if draw_map[current_node_id][i] == 1:
                        draw_map[current_node_id][i] = 0
                        draw_map[i][current_node_id] = 0
                break
            # 写个函数，用AOV网来解决一下，把边表和选择/并发群组传入，返回收敛到的结点id
            convergency_node, before_convergency_node = find_convergency_node(name, copy.deepcopy(relation_tag_matrix),
                                                     copy.deepcopy(current_level_nodes))
            act_1 = current_level_nodes.pop()
            act_2 = current_level_nodes.pop()
            # 选择组
            if relation_tag_matrix[act_1][act_2] == 3:
                g.node(name='before'+str(before_proxy_num), label='X', shape='diamond')
                g.node(name='after' + str(after_proxy_num), label='X', shape='diamond')

            # 并发组
            elif relation_tag_matrix[act_1][act_2] == 4:
                g.node(name='before' + str(before_proxy_num), label='+', shape='diamond')
                g.node(name='after' + str(after_proxy_num), label='+', shape='diamond')
            # 未明确指定，本应该默认并发，然而会出现三角错误，故分情况讨论
            else:
                # 收敛到自身
                if convergency_node == act_1 or convergency_node == act_2 or convergency_node in current_level_nodes:
                    g.node(name='before' + str(before_proxy_num), label='X', shape='diamond')
                    g.node(name='after' + str(after_proxy_num), label='X', shape='diamond')
                else:
                    g.node(name='before' + str(before_proxy_num), label='+', shape='diamond')
                    g.node(name='after' + str(after_proxy_num), label='+', shape='diamond')

            # 前代理连结点组，注意，结点组里面如果有一个是收敛节点，那么不连接该结点，而是连回收节点
            # 出现三角形
            is_triangle = False
            if convergency_node != act_1:
                g.edge('before' + str(before_proxy_num), str(act_1), color="#272727")
            else:
                is_triangle = True
            if convergency_node != act_2:
                g.edge('before' + str(before_proxy_num), str(act_2), color="#272727")
            else:
                is_triangle = True
            while len(current_level_nodes) > 0:
                act = current_level_nodes.pop()
                if convergency_node != act:
                    g.edge('before' + str(before_proxy_num), str(act), color="#272727")
                else:
                    is_triangle = True
            # 父节点连前代理
            g.edge(str(current_node_id), 'before' + str(before_proxy_num), color="#272727")
            for i in range(0, activity_num + 2):
                if draw_map[current_node_id][i] == 1:
                    draw_map[current_node_id][i] = 0
                    draw_map[i][current_node_id] = 0
            # 谁连后代理
            if is_triangle:# 有三角形形态
                g.edge('before' + str(before_proxy_num), 'after' + str(after_proxy_num), color="#272727")

            for node in before_convergency_node:
                node_to_agency[node] = 'after' + str(after_proxy_num)
                # g.edge(str(node), 'after' + str(after_proxy_num), color="#272727")
                draw_map[node][convergency_node] = 0
                draw_map[convergency_node][node] = 0
            # 后代理连收敛结点
            g.edge('after' + str(after_proxy_num), node_agency_former[convergency_node], color="#272727")
            node_agency_former[convergency_node] = 'after' + str(after_proxy_num)
            before_proxy_num += 1
            after_proxy_num += 1

            # 在群组里面保存一下处理过的群组
            nodes_and_before_agency[nodes_group] = 'before' + str(before_proxy_num - 1)

        is_finished.add(current_node_id)
        for i in range(0, activity_num+2):
            if relation_tag_matrix[current_node_id][i] == 1 and i not in is_finished:
                to_do_node.append(i)



    # 开始绘制后继关系
    have_edge = set()
    for i in range(0, activity_num+2):
        for j in range(0, activity_num+2):
            if draw_map[i][j] == 1 and str(i)+str(j) not in have_edge:
                g.edge(str(i), str(j), color="#272727")
                have_edge.add(str(i)+str(j))
    # 谁连后代理，处理
    for i in range(0, activity_num+2):
        if node_to_agency[i] != "-1":
            g.edge(str(i), node_to_agency[i], color="#272727")
    g.render(filename=name, directory="figure", view=False)

if __name__ == "__main__":
    # 文件路径规约
    file_path = '../data/'
    activity_temporal_relationship_file_name = 'tag_2.txt'
    activity_temporal_relationship_file_path = file_path + activity_temporal_relationship_file_name

    # [file_name1, file_name2...]
    semantic_roles_entity_file_list = []
    # {file_name:tag_seq}
    activity_temporal_relationship_and_file_name = {}
    # [file_name, [activity_list]]
    semantic_roles_entity_content_list = []

    for file in os.listdir(file_path):
        if file != 'tag_2.txt':
            semantic_roles_entity_file_list.append(file)

    #[file_name, content]处理文件名和活动列表
    for cur_file in semantic_roles_entity_file_list:
        # 提取活动列表
        activity_list = []
        content = []
        with open(file_path+cur_file, encoding='utf-8') as file:
            temp = file.read()
            temp = temp.split('\n')
            for line in temp:
                content.append(line.split())
        content = content[:-1]
        cur_activity_content = ''
        for pair in content:
            if pair[1] == 'B-activity':
                if cur_activity_content != '':
                    activity_list.append(cur_activity_content)
                    cur_activity_content = ''
                cur_activity_content += pair[0]
            elif pair[1] == 'I-activity':
                cur_activity_content += (' ' + pair[0])
            else:
                if cur_activity_content != '':
                    activity_list.append(cur_activity_content)
                    cur_activity_content = ''
        if cur_activity_content != '':
            activity_list.append(cur_activity_content)
            cur_activity_content = ''
        semantic_roles_entity_content_list.append([cur_file, activity_list])

    # 处理文件名和时序关系序列
    content = []
    with open(file_path + activity_temporal_relationship_file_name, encoding='utf-8') as file:
        temp = file.read()
        temp = temp.split('\n')
        for line in temp:
            content.append(line.split())
    content = content[:-1]
    for pair in content:
        activity_temporal_relationship_and_file_name[pair[0]] = pair[1]


    # 逐一绘图
    for pair in semantic_roles_entity_content_list:
        tag_seq = activity_temporal_relationship_and_file_name[pair[0]]
        name = pair[0][0:-10]
        activity_num = get_activity_num(tag_seq)
        activity_list = pair[1]
        activity_relationship_to_BPMN(name, activity_num, tag_seq, activity_list)


