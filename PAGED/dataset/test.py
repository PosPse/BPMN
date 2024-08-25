from graphviz import Digraph
from graph_utils import Graph

train_dataset = Graph.load_from_json('/home/btr/bpmn/PAGED/dataset/procedural_graph_extraction/train.json')
for data in train_dataset:
    print(data.file_index)
    