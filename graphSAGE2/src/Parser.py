import argparse

# 路径设置等（一般不用修改）
parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--name', type=str, default='debug')
# 随机数，据说最佳是3407
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--data_dir', type=str, default='/home/btr/bpmn/graphSAGE2/data/')
# 词表长度30524
parser.add_argument('--vocab_dir', type=str, default='/home/btr/bpmn/graphSAGE2/src/vocab.txt')
parser.add_argument('--vocab_len', type=int, default=30524)
parser.add_argument('--num_labels', type=int, default=5)
parser.add_argument('--b_sz', type=int, default=1)

# 有关主模型的设置
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--hidden_emb_size', type=int, default=128)
# LSTM BiLSTM Transformer RNN 四选1
parser.add_argument('--agg_func', type=str, default='LSTM')
parser.add_argument('--gcn', action='store_true')
# 此epoch有时为未取得最低loss的最大epoch
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.3)

# 关于语义融合函数的相关参数设置
parser.add_argument('--lstm_hidden_size', type=int, default=128)
parser.add_argument('--lstm_num_layers', type=int, default=1)
# transformer最大支持的len长度
parser.add_argument('--transformer_max_len', type=int, default=32)
parser.add_argument('--transformer_batch_num', type=int, default=1)
# transformer堆叠层数
parser.add_argument('--transformer_layers', type=int, default=4)
parser.add_argument('--transformer_hidden_size', type=int, default=128)
# transformer全连接层隐向量长度，一般默认为hidden_size的一半
parser.add_argument('--transformer_d_ff', type=int, default=512)
# 自注意力头的个数
parser.add_argument('--transformer_h', type=int, default=4)
parser.add_argument('--transformer_dropout', type=int, default=0.1)

args = parser.parse_args()