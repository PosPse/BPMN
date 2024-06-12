import argparse

parser = argparse.ArgumentParser(description="A simple LLM.")

parser.add_argument("--vocab_dir", type=str, default="/home/btr/bpmn/llmsage/my-data/vocab.txt", help="词表文件路径")
parser.add_argument("--vocab_len", type=int, default="30529", help="词表长度")
parser.add_argument("--embedding_size", type=int, default="128", help="词向量维度")

parser.add_argument("--datasets_json", type=str, default="/home/btr/bpmn/llmsage/my-data/datasets3.json", help="JSON格式的数据集路径")
parser.add_argument("--batch_size", type=int, default="1", help="批量大小")
parser.add_argument("--shuffle", type=bool, default="True", help="是否打乱数据")
args = parser.parse_args()