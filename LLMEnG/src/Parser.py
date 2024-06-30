import argparse

parser = argparse.ArgumentParser(description="A simple LLM.")

parser.add_argument("--vocab_dir", type=str, default="/home/btr/bpmn/LLMEnG/my-data/vocab.txt", help="词表文件路径")
parser.add_argument("--vocab_len", type=int, default="30529", help="词表长度")
parser.add_argument("--embedding_size", type=int, default="128", help="词向量维度")

parser.add_argument("--datasets_json", type=str, default="/home/btr/bpmn/LLMEnG/my-data/datasets3.json", help="JSON格式的数据集路径")
parser.add_argument("--batch_size", type=int, default="1", help="批量大小")
parser.add_argument("--shuffle", type=bool, default="False", help="是否打乱数据")

parser.add_argument("--hidden_size", type=int, default="128", help="隐藏层大小")
parser.add_argument("--node_num_classes", type=int, default="6", help="节点分类数量")
parser.add_argument("--aggr", type=str, default="lstm", help="聚合函数")


parser.add_argument("--lr", type=float, default="0.0001", help="学习率")
parser.add_argument("--epochs", type=int, default="200", help="训练轮数")
args = parser.parse_args()