import Transformer
from loss import *
from sklearn.utils import shuffle
import torch.nn as nn


def evaluate(dataCenter, graphSage, classification, agg_func_model, name, cur_epoch, is_test, zhe):
	test_nodes = getattr(dataCenter, 'test')
	val_nodes = getattr(dataCenter, 'val')
	train_nodes = getattr(dataCenter, 'train')
	labels = getattr(dataCenter, 'labels')
	total_data = getattr(dataCenter, 'total_data')
	models = [graphSage, classification, agg_func_model]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)
	predicts = []
	labels_val = []
	logists = torch.tensor([1.,1.,1.,1.,1.]).reshape(1,-1)
	for node in val_nodes[zhe]:
		emb = graphSage([node])
		logist = classification(emb, dataCenter, [node])
		_, predict = torch.max(logist, 1)
		label_val = [int(i) for i in labels[node][0]]
		assert len(label_val) == len(predict)
		for i in predict:
			predicts.append(i)
		for i in label_val:
			labels_val.append(i)
		logists = torch.cat([logists, logist], dim=0)
	logists = logists[1:]
	CrossEntropyLoss = Loss(dataCenter.rate[zhe][1])
	loss_val = CrossEntropyLoss(logists, labels_val)[0]
	loss_val /= len(labels_val)
	print('验证集损失:'+str(loss_val))
	if is_test:
		labels_test = []
		predicts = []
		logists = torch.tensor([1.,1.,1.,1.,1.]).reshape(1,-1)
		for node in test_nodes:
			emb = graphSage([node])
			logist = classification(emb, dataCenter, [node])
			_, predict = torch.max(logist, 1)
			label_test = [int(i) for i in labels[node][0]]
			assert len(label_test) == len(predict)
			labels_test.extend(label_test)
			predicts.extend(predict)
			logists = torch.cat([logists, logist], dim=0)
		logists = logists[1:]
		CrossEntropyLoss = Loss(dataCenter.rate[0][0])
		loss_test = CrossEntropyLoss(logists, labels_test)[0]
		loss_test /= len(labels_test)
		print('测试集损失:' + str(loss_test))
		labels_test = torch.tensor(labels_test)
		predicts = torch.tensor(predicts)
		evaluate_lwx(predicts, labels_test)
		torch.save(models, '../models/best_model.torch')
	'''
	if is_test:
		total_nodes = np.concatenate((train_nodes, val_nodes, test_nodes), axis=0)
		labels_total = []
		predicts = []
		logists = torch.tensor([1.,1.,1.,1.,1.]).reshape(1,-1)
		file = open('result.txt', mode='w')
		for node in total_nodes:
			emb = graphSage([node])
			logist = classification(emb, dataCenter, [node])
			_, predict = torch.max(logist, 1)
			cur_label = [int(i) for i in labels[node][0]]
			assert len(cur_label) == len(predict)
			labels_total.extend(cur_label)
			predicts.extend(predict)
			temp = predict.numpy().tolist()
			for i in temp: file.write(str(i))
			file.write('隔离')
			for i in cur_label: file.write(str(i))
			file.write('\n')
			logists = torch.cat([logists, logist], dim=0)
		file.close()
		logists = logists[1:]
		CrossEntropyLoss = Loss()
		loss_test = CrossEntropyLoss(logists, labels_total)[0]
		loss_test /= len(labels_total)
		print('所有数据损失:' + str(loss_test))
		labels_total = torch.tensor(labels_total)
		predicts = torch.tensor(predicts)
		print('所有数据:\n')
		evaluate_lwx(predicts, labels_total)
	'''
	for param in params:
		param.requires_grad = True


	return loss_val

def apply_model(dataCenter, graphSage, classification, agg_function_model, b_sz, lr, cur_epoch, zhe):
	train = getattr(dataCenter, 'train')
	labels = getattr(dataCenter, 'labels')

	train_nodes = shuffle(train)

	models = [graphSage, classification, agg_function_model] #似乎graphsage里面已经包含了agg_function_model
	# models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.SGD(params, lr=lr)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	total_loss = 0.
	CrossEntropyLoss = Loss(dataCenter.rate[zhe][0])

	for index in range(len(train_nodes[zhe])):
		nodes_id = [train_nodes[zhe][index]]
		labels_batch = labels[nodes_id[0]]
		embs_batch = graphSage(nodes_id)
		logists = classification(embs_batch, dataCenter, nodes_id)
		labels_batch = [int(i) for i in labels_batch[0]]
		if logists.shape[0] != len(labels_batch):
			print('取的数据和标签不匹配，节点id::'+str(nodes_id))
		loss_sup = CrossEntropyLoss(logists, labels_batch)[0]
		loss_sup /= len(labels_batch)
		loss = loss_sup
		total_loss += float(loss)
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()
		optimizer.zero_grad()
		for model in models:
			model.zero_grad()
	print('epoch [{}], Loss: {:.4f}'.format(cur_epoch, total_loss/len(train_nodes[zhe])))
	return graphSage, classification, total_loss/len(train_nodes[zhe])

def evaluate_lwx(predicts, labels):
	predicts = predicts.tolist()
	labels = labels.tolist()
	result_file = r'result.txt'
	file = open(result_file, 'w')

	class_0_prediction_right = 0
	class_0_prediction_false = 0
	class_1_prediction_right = 0
	class_1_prediction_false = 0
	class_2_prediction_right = 0
	class_2_prediction_false = 0
	class_3_prediction_right = 0
	class_3_prediction_false = 0
	class_4_prediction_right = 0
	class_4_prediction_false = 0
	#从预测文件角度分析
	for i in range(0, len(predicts)):
		if predicts[i] == 0:
			if predicts[i] == labels[i]:
				class_0_prediction_right += 1
			else:
				class_0_prediction_false += 1
		if predicts[i] == 1:
			if predicts[i] == labels[i]:
				class_1_prediction_right += 1
			else:
				class_1_prediction_false += 1
		if predicts[i] == 2:
			if predicts[i] == labels[i]:
				class_2_prediction_right += 1
			else:
				class_2_prediction_false += 1
		if predicts[i] == 3:
			if predicts[i] == labels[i]:
				class_3_prediction_right += 1
			else:
				class_3_prediction_false += 1
		if predicts[i] == 4:
			if predicts[i] == labels[i]:
				class_4_prediction_right += 1
			else:
				class_4_prediction_false += 1

	file.write(str(class_0_prediction_right) + ' ' + str(class_0_prediction_false) + '\n')
	file.write(str(class_1_prediction_right) + ' ' + str(class_1_prediction_false) + '\n')
	file.write(str(class_2_prediction_right) + ' ' + str(class_2_prediction_false) + '\n')
	file.write(str(class_3_prediction_right) + ' ' + str(class_3_prediction_false) + '\n')
	file.write(str(class_4_prediction_right) + ' ' + str(class_4_prediction_false) + '\n' + '\n')

	class_0_prediction_right = 0
	class_0_prediction_false = 0
	class_1_prediction_right = 0
	class_1_prediction_false = 0
	class_2_prediction_right = 0
	class_2_prediction_false = 0
	class_3_prediction_right = 0
	class_3_prediction_false = 0
	class_4_prediction_right = 0
	class_4_prediction_false = 0
	# 从标签文件角度分析
	for i in range(0, len(labels)):
		if labels[i] == 0:
			if predicts[i] == labels[i]:
				class_0_prediction_right += 1
			else:
				class_0_prediction_false += 1
		if labels[i] == 1:
			if predicts[i] == labels[i]:
				class_1_prediction_right += 1
			else:
				class_1_prediction_false += 1
		if labels[i] == 2:
			if predicts[i] == labels[i]:
				class_2_prediction_right += 1
			else:
				class_2_prediction_false += 1
		if labels[i] == 3:
			if predicts[i] == labels[i]:
				class_3_prediction_right += 1
			else:
				class_3_prediction_false += 1
		if labels[i] == 4:
			if predicts[i] == labels[i]:
				class_4_prediction_right += 1
			else:
				class_4_prediction_false += 1

	file.write(str(class_0_prediction_right) + ' ' + str(class_0_prediction_false) + '\n')
	file.write(str(class_1_prediction_right) + ' ' + str(class_1_prediction_false) + '\n')
	file.write(str(class_2_prediction_right) + ' ' + str(class_2_prediction_false) + '\n')
	file.write(str(class_3_prediction_right) + ' ' + str(class_3_prediction_false) + '\n')
	file.write(str(class_4_prediction_right) + ' ' + str(class_4_prediction_false) + '\n' + '\n')

#把激活函数做成不固定的，get一个激活函数
def get_agg_func(args, Data_center):
	if args.agg_func == 'LSTM':
		return torch.nn.LSTM(input_size=args.hidden_emb_size, hidden_size=args.lstm_hidden_size,
						 num_layers=args.lstm_num_layers, batch_first=True)
	elif args.agg_func == 'BiLSTM':
		return torch.nn.LSTM(input_size=args.hidden_emb_size, hidden_size=args.lstm_hidden_size,
						 num_layers=args.lstm_num_layers, batch_first=True, bidirectional=True, dropout=0)
	elif args.agg_func == 'Transformer':
		return Transformer.make_model(max_len=args.transformer_max_len, batch_num=args.transformer_batch_num,
									  N=args.transformer_layers, d_model=args.transformer_hidden_size,
									  d_ff=args.transformer_d_ff, h=args.transformer_h, dropout=args.transformer_dropout,
									  Data_center=Data_center)
	elif args.agg_func == 'RNN':
		pass



