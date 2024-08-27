from graph_utils import Graph, DrawUtil
from tqdm import tqdm
train_dataset = Graph.load_from_json('/home/btr/bpmn/PAGED/dataset/procedural_graph_extraction/train.json')
d_util = DrawUtil(save_path='/home/btr/bpmn/PAGED/dataset/image/train')
for data in tqdm(train_dataset):
    dot = d_util.draw(data)
    d_util.save(dot)