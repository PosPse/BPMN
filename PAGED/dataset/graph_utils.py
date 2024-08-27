import json
from enum import Enum
from graphviz import Digraph
import textwrap

path = '/home/btr/bpmn/PAGED/dataset/BPMN_Icon/'

class NodeImage(Enum):
    StartNode = f'{path}BpmnStartEvent.png'
    EndNode = f'{path}BpmnEndEvent.png'
    AND = f'{path}BpmnGatewayParallel.png'
    OR = f'{path}BpmnGatewayOr.png'
    XOR = f'{path}BpmnGatewayXor.png'
    Activity = ''
    DataObject = f'{path}BpmnDataObject.png'
    TextAnnotation = f'{path}BpmnTextAnnotation.png'

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
        self.step_nodes = [Node(node) for node in data['step_nodes']] if data['step_nodes'] != [] else []
        self.data_nodes = [Node(node) for node in data['data_nodes']] if data['data_nodes'] != [] else []
        self.text_nodes = [Node(node) for node in data['text_nodes']] if data['text_nodes'] != [] else []
        self.SequenceFlow = [SequenceFlow(flow) for flow in data['SequenceFlow']]
        self.MessageFlow = [Flow(flow) for flow in data['MessageFlow']] if data['MessageFlow'] != [] else []
        self.Association = [Flow(flow) for flow in data['Association']] if data['Association'] != [] else []
        self.or_node_num = data['or_node_num']
        self.agent_set = self.get_agent_set()
    
    def get_agent_set(self):
        setp_node_agent_set = set([node.agent for node in self.step_nodes])
        data_node_agent_set = set([node.agent for node in self.data_nodes])
        text_node_agent_set = set([node.agent for node in self.text_nodes])
        agent_set = setp_node_agent_set.union(data_node_agent_set).union(text_node_agent_set)
        agent_set = list(agent_set)
        agent_set = [agent for agent in agent_set if agent != '']
        return agent_set

    @staticmethod
    def load_from_json(file: str):
        with open(file, 'r') as f:
            dataset = json.load(f)
            if dataset:
                dataset = [Graph(data) for data in dataset]
                return dataset
            else:
                return None

class DrawUtil:
    def __init__(self, save_path, format='PDF', cleanup=True):
        self.format = format
        self.save_path = save_path
        self.cleanup = cleanup
    
    def wrap_text(self, sentence, width=35):
        wrapped_text = textwrap.fill(sentence, width)
        return wrapped_text

    def draw_start_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label='Start' if label == '' else label, penwidth='0', labelloc='b')
            sub.node(id, label='',fixedsize='true',penwidth='0', shapefile=NodeImage.StartNode.value, imagescale='both', width='0.8', height='0.8')
    
    def draw_end_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label='End' if label == '' else label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.EndNode.value, imagescale='both', width='0.8', height='0.8')
    
    def draw_and_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label=label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.AND.value, imagescale='both', width='0.8', height='0.8')

    def draw_or_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label=label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.OR.value, imagescale='both', width='0.8', height='0.8')

    def draw_xor_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label=label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.XOR.value, imagescale='both', width='0.8', height='0.8')

    def draw_activity_node(self, dot, id, label):
        dot.node(id, shape='rect', style='rounded,bold', label=label)


    def draw_data_object_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label=label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.DataObject.value, imagescale='both', width='0.8', height='0.8')

    def draw_text_annotation_node(self, dot, id, label):
        with dot.subgraph(name=f'cluster_{id}') as sub:
            sub.attr(label=label, penwidth='0', labelloc='b')
            sub.node(id, label='', fixedsize='true',penwidth='0', shapefile=NodeImage.TextAnnotation.value, imagescale='both', width='0.8', height='0.8')

    def draw_node(self, dot, node_list):
        for node in node_list:
            if node.type == 'StartNode':
                self.draw_start_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'EndNode':
                self.draw_end_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'AND':
                self.draw_and_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'OR':
                self.draw_or_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'XOR':
                self.draw_xor_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'Activity':
                self.draw_activity_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'DataObject':
                self.draw_data_object_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'TextAnnotation':
                self.draw_text_annotation_node(dot, node.resourceId, self.wrap_text(node.NodeText))
            else:
                pass
    
    def draw_node_with_swim_lane(self, dot, node_list):
        for node in node_list:
            if node.type == 'StartNode':
                if node.agent == '':
                    self.draw_start_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_start_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'EndNode':
                if node.agent == '':
                    self.draw_end_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_end_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'AND':
                if node.agent == '':
                    self.draw_and_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_and_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'OR':
                if node.agent == '':
                    self.draw_or_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_or_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'XOR':
                if node.agent == '':
                    self.draw_xor_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_xor_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'Activity':
                if node.agent == '':
                    self.draw_activity_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_activity_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'DataObject':
                if node.agent == '':
                    self.draw_data_object_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_data_object_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            elif node.type == 'TextAnnotation':
                if node.agent == '':
                    self.draw_text_annotation_node(dot, node.resourceId, self.wrap_text(node.NodeText))
                else:
                    with dot.subgraph(name=f'cluster_{node.agent}') as sub:
                        sub.attr(label=node.agent, labelloc='b')
                        self.draw_text_annotation_node(sub, node.resourceId, self.wrap_text(node.NodeText))
            else:
                pass
    def draw_flow(self, dot, flow_list):
        for flow in flow_list:
            dot.edge(flow.src, flow.tgt, label=flow.condition if hasattr(flow, 'condition') else '')
    def draw(self, graph):
        title = graph.paragraph
        title = self.wrap_text(title, 100)
        dot = Digraph(name=graph.file_index, format=self.format)
        dot.attr('graph', label=title)
        # dot.attr('graph', rankdir='LR')
        if graph.agent_set != []:
            self.draw_node_with_swim_lane(dot, graph.step_nodes)
            self.draw_node_with_swim_lane(dot, graph.data_nodes)
            self.draw_node_with_swim_lane(dot, graph.text_nodes)
            self.draw_flow(dot, graph.SequenceFlow)
            self.draw_flow(dot, graph.MessageFlow)
            self.draw_flow(dot, graph.Association)
        else:
            self.draw_node(dot, graph.step_nodes)
            self.draw_node(dot, graph.data_nodes)
            self.draw_node(dot, graph.text_nodes)
            self.draw_flow(dot, graph.SequenceFlow)
            self.draw_flow(dot, graph.MessageFlow)
            self.draw_flow(dot, graph.Association)
        return dot
    
    def save(self, dot):
        dot.render(filename=dot.name, directory=self.save_path, cleanup=self.cleanup)
                            