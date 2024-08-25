import json

class Node:
    def __init__(self, node: dict):
        self.resourceId = node['resourceId']
        self.NodeText = node['NodeText']
        self.agent = node['agent']
        self.type = node['type']

class SequenceFlow:
    def __init__(self, flow: dict):
        self.src = flow['src']
        self.tgt = flow['tgt']
        self.condition = flow['condition']

class Flow:
    def __init__(self, flow: dict):
        self.src = flow['src']
        self.tgt = flow['tgt']

class Graph:
    def __init__(self, data:dict):
        self.file_index = str(data['file_index'])
        self.paragraph = data['paragraph']
        self.step_nodes = [Node(node) for node in data['step_nodes']] if data['step_nodes'] != [] else None
        self.data_nodes = [Node(node) for node in data['data_nodes']] if data['data_nodes'] != [] else None
        self.text_nodes = [Node(node) for node in data['text_nodes']] if data['text_nodes'] != [] else None
        self.SequenceFlow = [SequenceFlow(flow) for flow in data['SequenceFlow']]
        self.MessageFlow = [Flow(flow) for flow in data['MessageFlow']] if data['MessageFlow'] != [] else None
        self.Association = [Flow(flow) for flow in data['Association']] if data['Association'] != [] else None
        self.or_node_num = data['or_node_num']

    @staticmethod
    def load_from_json(file: str):
        with open(file, 'r') as f:
            dataset = json.load(f)
            if dataset:
                dataset = [Graph(data) for data in dataset]
                return dataset
            else:
                return None
            
a = Graph.load_from_json('/home/btr/bpmn/PAGED/dataset/procedural_graph_extraction/train.json')
print(a)