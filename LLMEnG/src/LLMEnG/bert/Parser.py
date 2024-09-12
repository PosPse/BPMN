import argparse

parser = argparse.ArgumentParser(description="A simple LLM.")
'''
    通用参数
'''
parser.add_argument("--device", type=str, default="cuda:1", choices=['cpu', 'cuda:0', 'cuda:1'], help="GPU设备")
# 数据
parser.add_argument("--datasets_json", type=str, default="/home/user/baitianrui/bpmn/LLMEnG/my-data/datasets8-llama3.1-70B.json", help="JSON格式的数据集路径")
parser.add_argument("--batch_size", type=int, default="1", help="批量大小")
parser.add_argument("--shuffle", type=bool, default="False", help="是否打乱数据")
# 模型
parser.add_argument("--node_num_classes", type=int, default="6", help="节点分类数量")
parser.add_argument("--edge_num_classes", type=int, default="11", help="边分类数量")
parser.add_argument("--fusion_method", type=str, default="concat", choices=['concat', 'attention'], help="边特征融合方法")
# 训练
parser.add_argument("--hidden_size", type=int, default="128", help="隐藏层大小")
parser.add_argument("--aggr", type=str, default="mean", choices=['mean', 'lstm'], help="GraphSAGE聚合函数")
parser.add_argument("--lr", type=float, default="0.01", help="学习率")
parser.add_argument("--epochs", type=int, default="180", help="训练轮数")

# token embedding
parser.add_argument("--llm_model", type=str, default="/home/user/baitianrui/bpmn/model/safetensors/bert-base-uncased", help="LLM模型路径")
parser.add_argument("--the_way_of_emb_new_token", type=str, default="mean", choices=['random', 'mean', 'zero'], help="新token的嵌入方式: random | mean | zero | copy")
parser.add_argument("--the_way_of_token_emb", type=str, default="sum", choices=['cls'], help="token嵌入的获取方式: cls(使用CLS作为token嵌入) | mean(使用最后一层模型输出，求平均) | pooler_output(bert模型的输出)")

# 方法
parser.add_argument("--method", type=str, default="add", choices=['none', 'add', 'del', 'add_del'], help="edge_index构建方法: none | add | del | add_del")
parser.add_argument("--add_scale", type=float, default="0.1", help="add方法的比例")
parser.add_argument("--del_scale", type=float, default="0.1", help="del方法的比例")
args = parser.parse_args()