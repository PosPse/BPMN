import sys
sys.path.append('/root/anaconda3/envs/lwx/lib/python3.8/site-packages')
sys.path.append('/root/anaconda3/envs/lwx/lib/python38.zip')
sys.path.append('/root/anaconda3/envs/lwx/lib/python3.8')
sys.path.append('/root/anaconda3/envs/lwx/lib/python3.8/lib-dynload')
sys.path.append('/root/lwx/graphSAGE2/src')
sys.path.append('/root/lwx/graphSAGE2')
sys.path.append('/root/lwx')
print(sys.path)
from dataCenter import *
from utils import *
from models import *
import Parser


# sys.path.append('/home/zhurui/.pycharm_helpers/pycharm_display')
# sys.path.append('/home/zhurui/.local/lib/python3.8/site-packages')
# sys.path.append('/home/zhurui/.pycharm_helpers/pycharm_matplotlib_backend')

def train_func():
	print(torch.__version__)
	# 导入参数配置文件
	
	args = Parser.args
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# 加载数据
	dataCenter = DataCenter(args.data_dir, args.vocab_dir, args.embedding_size, args.vocab_len)
	dataCenter.load_dataSet()
	features = []
	all_feature = getattr(dataCenter, 'feats')
	for feature in all_feature:
		features.append(torch.FloatTensor(feature))

	# 开始初始化模型
	agg_func = get_agg_func(args=args, Data_center=dataCenter)
	graphSage = GraphSage(args.num_layers, args.embedding_size, args.hidden_emb_size, features,
						  getattr(dataCenter, 'adj_lists'), agg_function_model=agg_func,
						  data_center=dataCenter, gcn=args.gcn, agg_func=args.agg_func)
	classification = Classification(args.hidden_emb_size*2, args.num_labels)

	cur_epoch = 1
	num = 0
	best_loss_val = 10000
	while num < args.epochs:
		dataCenter.reget_data()
		print('----------------------EPOCH %d-----------------------' % cur_epoch)
		# K折交叉验证，循环执行K次
		train_nodes = getattr(dataCenter, 'train')
		loss_train = 0.
		for i in range(0, len(train_nodes)):
			graphSage, classification, loss_train = apply_model(dataCenter, graphSage, classification, agg_func,
													 args.b_sz, args.lr, cur_epoch, i)
			loss_val = evaluate(dataCenter, graphSage, classification, agg_func, args.name, cur_epoch, False, i)
			loss_train += loss_val
		if loss_train < best_loss_val:
			best_loss_val = loss_train
			num = 1
		elif loss_train >= best_loss_val:
			num += 1
		cur_epoch += 1
	print("终于训练完啦！")
	evaluate(dataCenter, graphSage, classification, agg_func, args.name, cur_epoch, True, 0)

if __name__ == '__main__':
	train_func()