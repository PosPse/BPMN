from graph_utils import Graph, DrawUtil
from tqdm import tqdm
train_dataset = Graph.load_from_json('/home/btr/bpmn/PAGED/dataset/procedural_graph_extraction/dev.json')
d_util = DrawUtil(save_path='/home/btr/bpmn/PAGED/dataset/image/dev')
for data in tqdm(train_dataset):
    dot = d_util.draw(data)
    d_util.save(dot)