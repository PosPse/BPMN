import argparse

parser = argparse.ArgumentParser(description="A simple LLM.")

parser.add_argument("--vocab_dir", type=str, default="/home/btr/bpmn/llmsage/my-data/vocab.txt", help="Path to the vocabulary file.")
parser.add_argument("--vocab_len", type=int, default="30529", help="Length of the vocabulary.")
parser.add_argument("--embedding_size", type=int, default="128", help="Size of the embedding.")

parser.add_argument("--datasets_json", type=str, default="/home/btr/bpmn/llmsage/my-data/datasets3.json", help="Path to the data file.")

args = parser.parse_args()