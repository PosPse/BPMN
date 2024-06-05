import sys
sys.path.append('/home/btr/bpmn/llmsage/src')

from data_utils import DataCenter
import Parser

args = Parser.args

data_center = DataCenter(datasets_json=args.datasets_json, vocab_dir=args.vocab_dir, vocab_len=args.vocab_len, embedding_size=args.embedding_size)
