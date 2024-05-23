import os
from collections import defaultdict
import numpy as np
import re
import torch
import math
import random

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, data_dir, vocab_dir, embedding_size, vocab_len):
		super(DataCenter, self).__init__()
		self.data_dir = data_dir
		self.vocab_dir = vocab_dir
		self.embedding_size = embedding_size
		self.embed = torch.nn.Embedding(vocab_len, self.embedding_size)
	def load_dataSet(self):
		"""加载数据集
			self.total_data: [
								[['44_data1.txt'], 
								['[activity]', 'when', '[activity]', 'once', '[activity]', '[activity]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', ...], 
								['210000000000000000000001000000000100000000111000003300010300010100001000100101'], 
								array([[-0.5401893 ,  2.0286763 ,  0.28921074, ...,  1.7957834 ,
									-0.8011223 ,  0.3343649 ],
									[ 0.7256031 ,  1.2615494 ,  1.5398114 , ..., -0.17598295,
										-1.0770534 ,  2.5428627 ],
									[ 0.36910808,  0.6125294 ,  1.276257  , ...,  1.7957834 ,
										-0.80089134,  0.3343649 ],
									...,
									[-1.3035511 , -1.1519649 ,  0.8526652 , ..., -0.00694638,
										0.65504575, -0.10282433],
									[ 0.13840985,  0.5091666 ,  0.4160018 , ...,  0.6382996 ,
										-1.0312877 ,  1.2081577 ],
									[-0.944227  ,  1.9434186 ,  1.2792368 , ...,  1.7957749 ,
										-0.7975425 ,  0.3343585 ]], dtype=float32),
								defaultdict(<class 'set'>, {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 5}, 5: {4, 6}, 6: {8, 5, 7}, 7: {9, 6}, 8: {9, 6}, 9: {8, 10, 11, 7}, 10: {9, 12}, 11: {9, 12}, 12: {10, 11, 13, 14}, 13: {12, 15}, 14: {12, 15}, 15: {16, 17, 13, 14}, 16: {18, 15}, 17: {18, 15}, 18: {16, 17, 19}, 19: {18, 20}, 20: {19, 21}, 21: {24, 20, 22, 23}, 22: {25, 21}, 23: {25, 21}, 24: {25, 21}, 25: {22, 23, 24, 26, 27, 28, 29, 30}, 26: {25, 31}, 27: {25, 31}, 28: {25, 31}, 29: {25, 31}, 30: {25, 31}, 31: {26, 27, 28, 29, 30}})
								]
								...
							]
			self.test: (13, )
			self.val: (5, 11)
			self.train: (5, 44)
			self.feats: self.total_data[i][3]
			self.labels: self.total_data[i][2]
			self.adj_lists: self.total_data[i][4]
			self.rate: 
		"""
		print("加载训练数据...")
		feat_data = []
		labels = []
		adj_lists = []
		# 获取原始数据
		file_list, tag_list = self.get_data()
		file_mask = self.data_2_mask(file_list)
		self.total_data = self.data_pack(file_mask, tag_list)
		self.total_data = self.add_feature(self.total_data)
		self.total_data = self.add_adj(self.total_data)
		for i in range(0, len(self.total_data)):
			labels.append(self.total_data[i][2])
			feat_data.append(self.total_data[i][3])
			adj_lists.append(self.total_data[i][4])
		test_indexs, val_indexs, train_indexs = self._split_data()
		setattr(self, 'total_data', self.total_data)
		setattr(self, 'test', test_indexs)
		setattr(self, 'val', val_indexs)
		setattr(self, 'train', train_indexs)
		setattr(self, 'feats', feat_data)
		setattr(self, 'labels', labels)
		setattr(self, 'adj_lists', adj_lists)
		rate = self.weight_calculate(train_indexs, val_indexs, self.total_data)
		setattr(self, 'rate', rate)
		print("训练数据加载完毕")

	def _split_data(self):
		"""划分数据集：

		Returns:
			test_indexs2: 
			val_indexs2: 
			train_indexs2: 
		"""
		'''
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		'''
		# 目前使用交叉验证
		train_and_val_indexs = [2,4,5,6,8,10,11,12,13,14,15,16,18,19,20,21,23,
								24,26,27,28,29,30,32,35,36,38,39,40,41,43,44,
								46,47,48,49,50,51,52,53,54,55,56,59,60,61,62,
								63,64,65,66,67,68,69,71]
		test_indexs = [1, 3, 7, 9, 17, 22, 25, 34, 37, 42, 45, 57, 58]
		#文档id和程序里文件的id不是一回事，需要重新映射
		train_and_val_indexs2 = []
		test_indexs2 = []
		train_indexs2 = []
		val_indexs2 = []

		for index in range(0, len(train_and_val_indexs)):
			file_name = str(train_and_val_indexs[index]) + '_data1.txt'
			for n in range(0, len(self.total_data)):
				if (self.total_data[n][0][0] == file_name):
					train_and_val_indexs2.append(n)

		for index in range(0, len(test_indexs)):
			file_name = str(test_indexs[index]) + '_data1.txt'
			for n in range(0, len(self.total_data)):
				if (self.total_data[n][0][0] == file_name):
					test_indexs2.append(n)

		random.shuffle(train_and_val_indexs2)

		begin = 0
		for i in range(0, 5):
			val_indexs2.append(train_and_val_indexs2[begin: begin+11])
			train_indexs2.append([])
			if begin != 0:
				train_indexs2[-1].extend(train_and_val_indexs2[0: begin])
			if begin != 44:
				train_indexs2[-1].extend(train_and_val_indexs2[begin+11:])
			begin += 11

		return np.array(test_indexs2), np.array(val_indexs2), np.array(train_indexs2)

	def get_data(self):
		"""从文件读取原始数据

		Returns:
			file_list: [
						['44_data1.txt', ['If', 'B-signal'], ['after', 'B-condition'], ['another', 'I-condition']]
						...]
				
			tag_list:  [
						['1_data1.txt', '110000300001100311010']
						...]
		"""
		data_files = os.listdir(self.data_dir)

		tag = ['I-activity', 'B-activity', 'B-signal', 'I-signal', 'O', 'punctuation', 'B-condition', 'I-condition']
		file_list = []
		tag_list = []
		for data_file in data_files:
			data_list = []
			if data_file != 'tag_2.txt':
				data_list.append(data_file)
				with open(self.data_dir + data_file, 'r', encoding='utf-8') as reader:
					for line in reader:
						line = re.sub('\n', '', line)
						temp = line.split(" ")
						# 检验数据标注的正确性
						if temp[1] not in tag:
							print("警告：出现未知tag:" + data_file + ':' + temp[0] + temp[1])
						data_list.append(temp)
				file_list.append(data_list)
			if data_file == 'tag_2.txt':
				with open(self.data_dir + data_file, 'r', encoding='utf-8') as reader:
					for line in reader:
						line = re.sub('\n', '', line)
						temp = line.split(" ")
						tag_list.append(temp)
		return file_list, tag_list

	def data_2_mask(self, file_list):
		"""将活动，条件等合并，保留信号词

		Args:
			file_list (_type_): file_list: [
											['44_data1.txt', ['If', 'B-signal'], ['after', 'B-condition'], ['another', 'I-condition']]
											...]

		Returns:
			file_mask: [
						['44_data1.txt', '[activity]', 'when', '[activity]', 'Once', '[activity]']
						...]
		"""
		result_list = []
		for i in range(0, len(file_list)):
			temp = file_list[i]
			para_temp = []
			for couple in temp:
				if couple[1] == 'B-activity':
					para_temp.append('[activity]')
				elif couple[1] == 'I-activity':
					pass
				elif couple[1] in ['B-signal', 'I-signal']:
					para_temp.append(couple[0])
				elif couple[1] in ['O']:
					pass
				elif couple[1] == 'B-condition':
					para_temp.append('[condition]')
				elif couple[1] == 'I-condition':
					pass
				elif couple[1] in ['punctuation']:
					pass
				else:
					para_temp.append(couple)
			result_list.append(para_temp)
		return result_list

	def data_pack(self, file_mask, tag_list):
		"""file_mask与tag_list合并

		Args:
			file_mask (_type_): [
								 ['44_data1.txt', '[activity]', 'when', '[activity]', 'Once', '[activity]']
								 ...]
			tag_list (_type_): [
								['1_data1.txt', '110000300001100311010']
								...]

		Returns:
				data_pack: [
							[['44_data1.txt'], ['[activity]', 'when', '[activity]', 'Once', '[activity]'],['210000']]
							...]
		"""
		total_data = []
		for file in file_mask:
			file_temp = []
			file_temp.append([file[0]])
			file_temp.append([])
			for i in range(1, len(file)):
				file_temp[1].append(file[i])
			for i in range(0,len(tag_list)):
				if tag_list[i][0] == file_temp[0][0]:
					file_temp.append([tag_list[i][1]])
			total_data.append(file_temp)
		return total_data

	def add_feature(self, total_data):
		"""添加feature：每个样本包括：文件名、data_2_mask、tag、data_2_mask中每个token128维

		Args:
			total_data (_type_): data_pack

		Returns:
			_type_: [
						[['44_data1.txt'], 
						['[activity]', 'when', '[activity]', 'once', '[activity]', '[activity]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', ...], 
						['210000000000000000000001000000000100000000111000003300010300010100001000100101'], 
						array([[-0.5401893 ,  2.0286763 ,  0.28921074, ...,  1.7957834 ,
							-0.8011223 ,  0.3343649 ],
							[ 0.7256031 ,  1.2615494 ,  1.5398114 , ..., -0.17598295,
								-1.0770534 ,  2.5428627 ],
							[ 0.36910808,  0.6125294 ,  1.276257  , ...,  1.7957834 ,
								-0.80089134,  0.3343649 ],
							...,
							[-1.3035511 , -1.1519649 ,  0.8526652 , ..., -0.00694638,
								0.65504575, -0.10282433],
							[ 0.13840985,  0.5091666 ,  0.4160018 , ...,  0.6382996 ,
								-1.0312877 ,  1.2081577 ],
							[-0.944227  ,  1.9434186 ,  1.2792368 , ...,  1.7957749 ,
								-0.7975425 ,  0.3343585 ]], dtype=float32)
						]
						...
					]
		"""
		for i in range(0, len(total_data)):
			# cur_content: ['[activity]', 'when', '[activity]', 'Once', '[activity]']
			cur_content = total_data[i][1] 
			# total_data[i][0][0]: '44_data1.txt'
			cur_id = self.token2id(cur_content, total_data[i][0][0])
			cur_feature = self.id2embedding(cur_id)
			total_data[i].append(cur_feature)
		return total_data
	
	def token2id(self, tokens, file_name):
		"""tokens中每个token在vocab.txt中的位置

		Args:
			tokens (file_mask): ['[activity]', 'when', '[activity]', 'Once', '[activity]']
			file_name (_type_): 44_data1.txt

		Returns:
			cur_id: [1, 2045, 1, 2322, 1, 1, 1, 2067, 2, 1, 2067, 2, 1, 2067, 2, 1, 2067, 2, 1, 2046, 1, 1, 2001, 2153, 2555, 1, 2001, 2153, 1999, 1998, 3574, 1]
		"""
		# 此处不能用集合，否则无法返回对应的index
		vocab_list = []
		result = []
		# 先变小写
		for i in range(0, len(tokens)):
			if tokens[i] != '[PAD]':
				tokens[i] = tokens[i].casefold()
		with open(self.vocab_dir, 'r', encoding='utf-8') as reader:
			for line in reader:
				line = re.sub('\n', '', line)
				vocab_list.append(line)
		for token in tokens:
			if token in vocab_list:
				result.append(vocab_list.index(token))
			else:
				print(token + "不在词表内,文件名:" + file_name)
				result.append(vocab_list.index('[UNK]'))
		return result

	def id2embedding(self, ids, is_numpy = True):
		""" 将ids中每个token嵌入到128维度，并转换为位置

		Args:
			ids (_type_): [1, 2045, 1, 2322, 1, 1, 1, 2067, 2, 1, 2067, 2, 1, 2067, 2, 1, 2067, 2, 1, 2046, 1, 1, 2001, 2153, 2555, 1, 2001, 2153, 1999, 1998, 3574, 1]
			is_numpy (bool, optional): 结果是否转换为numpy

		Returns:
			_type_: shape: len(ids) * 128
		"""
		ids = torch.LongTensor(ids)
		# 把每个token映射到128维
		result = self.embed(ids)
		result = self.embedding2positional(result, self.embedding_size)
		if(is_numpy):
			result = result.detach().numpy()
		return result

	def add_adj(self, total_data):
		"""添加活动的链接关系

		Args:
			total_data (_type_): add_feature

		Returns: 
			_type_:	[
						[['44_data1.txt'], 
						['[activity]', 'when', '[activity]', 'once', '[activity]', '[activity]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', 'if', '[condition]', '[activity]', ...], 
						['210000000000000000000001000000000100000000111000003300010300010100001000100101'], 
						array([[-0.5401893 ,  2.0286763 ,  0.28921074, ...,  1.7957834 ,
							-0.8011223 ,  0.3343649 ],
							[ 0.7256031 ,  1.2615494 ,  1.5398114 , ..., -0.17598295,
								-1.0770534 ,  2.5428627 ],
							[ 0.36910808,  0.6125294 ,  1.276257  , ...,  1.7957834 ,
								-0.80089134,  0.3343649 ],
							...,
							[-1.3035511 , -1.1519649 ,  0.8526652 , ..., -0.00694638,
								0.65504575, -0.10282433],
							[ 0.13840985,  0.5091666 ,  0.4160018 , ...,  0.6382996 ,
								-1.0312877 ,  1.2081577 ],
							[-0.944227  ,  1.9434186 ,  1.2792368 , ...,  1.7957749 ,
								-0.7975425 ,  0.3343585 ]], dtype=float32),
						defaultdict(<class 'set'>, {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 5}, 5: {4, 6}, 6: {8, 5, 7}, 7: {9, 6}, 8: {9, 6}, 9: {8, 10, 11, 7}, 10: {9, 12}, 11: {9, 12}, 12: {10, 11, 13, 14}, 13: {12, 15}, 14: {12, 15}, 15: {16, 17, 13, 14}, 16: {18, 15}, 17: {18, 15}, 18: {16, 17, 19}, 19: {18, 20}, 20: {19, 21}, 21: {24, 20, 22, 23}, 22: {25, 21}, 23: {25, 21}, 24: {25, 21}, 25: {22, 23, 24, 26, 27, 28, 29, 30}, 26: {25, 31}, 27: {25, 31}, 28: {25, 31}, 29: {25, 31}, 30: {25, 31}, 31: {26, 27, 28, 29, 30}})
						]
						...
					]
		"""
		for i in range(0,len(total_data)):
			cur_tokens = total_data[i][1]
			cur_no_activity = []
			last_activity = -1
			cur_adj_lists = defaultdict(set)
			for j in range(0, len(cur_tokens)):
				if cur_tokens[j] != '[activity]':
					cur_no_activity.append(j)
					if last_activity != -1:
						cur_adj_lists[last_activity].add(j)
						cur_adj_lists[j].add(last_activity)
				elif cur_tokens[j] == '[activity]' and len(cur_no_activity) != 0:
					for k in cur_no_activity:
						cur_adj_lists[k].add(j)
						cur_adj_lists[j].add(k)
					last_activity = j
					cur_no_activity = []
				elif cur_tokens[j] == '[activity]' and len(cur_no_activity) == 0:
					#两个activity直接相连的时候，或许可以考虑在中间加个什么标识
					if last_activity != -1:
						cur_adj_lists[last_activity].add(j)
						cur_adj_lists[j].add(last_activity)
					last_activity = j
			total_data[i].append(cur_adj_lists)
		return total_data

	def embedding2positional(self, embeddings, embedding_size, max_len=128):
		"""将ids中每个128维的token转换为位置

		Args:
			embeddings (_type_): shape: len(ids) * 128
			embedding_size (_type_): 128
			max_len (int, optional): _description_. Defaults to 128.

		Returns:
			_type_: len(ids) * 128
		"""
		pe = torch.zeros(max_len, embedding_size)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		embeddings = embeddings + pe[:embeddings.size(0)]
		return embeddings

	def  weight_calculate(self, train_indexs, val_indexs, total_data):
		rate = []
		train_indexs = list(train_indexs)
		val_indexs = list(val_indexs)
		if(len(train_indexs) == len(val_indexs)):
			print('训练集和验证集的折数一致，接下来进行动态权重加载...')
		else:
			print('训练集和验证集的折数不一致，程序退出...')
			return 0
		for lwx in range(0, len(train_indexs)):
			# K折交叉验证，为当前折加一个新列表
			rate.append([])
			# 为当前折的训练集加一个新列表
			rate[-1].append([])
			num_0 = 0.
			num_1 = 0.
			num_2 = 0.
			num_3 = 0.
			num_4 = 0.
			for index in train_indexs[lwx]:
				for i in range(0, len(total_data[index][2][0])):
					if total_data[index][2][0][i] == '0':
						num_0 += 1.
					if total_data[index][2][0][i] == '1':
						num_1 += 1.
					if total_data[index][2][0][i] == '2':
						num_2 += 1.
					if total_data[index][2][0][i] == '3':
						num_3 += 1.
					if total_data[index][2][0][i] == '4':
						num_4 += 1.
			total = num_0 + num_1 + num_2 + num_3 + num_4
			if num_0 > 0.9:
				rate[-1][0].append(total / num_0)
			else:
				rate[-1][0].append(0.)
			if num_1 > 0.9:
				rate[-1][0].append(total / num_1)
			else:
				rate[-1][0].append(0.)
			if num_2 > 0.9:
				rate[-1][0].append(total / num_2)
			else:
				rate[-1][0].append(0.)
			if num_3 > 0.9:
				rate[-1][0].append(total / num_3)
			else:
				rate[-1][0].append(0.)
			if num_4 > 0.9:
				rate[-1][0].append(total / num_4)
			else:
				rate[-1][0].append(0.)
			# 为当前折的验证集加一个新列表
			rate[-1].append([])
			num_0 = 0.
			num_1 = 0.
			num_2 = 0.
			num_3 = 0.
			num_4 = 0.
			for index in val_indexs[lwx]:
				for i in range(0, len(total_data[index][2][0])):
					if total_data[index][2][0][i] == '0':
						num_0 += 1.
					if total_data[index][2][0][i] == '1':
						num_1 += 1.
					if total_data[index][2][0][i] == '2':
						num_2 += 1.
					if total_data[index][2][0][i] == '3':
						num_3 += 1.
					if total_data[index][2][0][i] == '4':
						num_4 += 1.
			total = num_0 + num_1 + num_2 + num_3 + num_4
			if num_0 > 0.9:
				rate[-1][1].append(total / num_0)
			else:
				rate[-1][1].append(0.)
			if num_1 > 0.9:
				rate[-1][1].append(total / num_1)
			else:
				rate[-1][1].append(0.)
			if num_2 > 0.9:
				rate[-1][1].append(total / num_2)
			else:
				rate[-1][1].append(0.)
			if num_3 > 0.9:
				rate[-1][1].append(total / num_3)
			else:
				rate[-1][1].append(0.)
			if num_4 > 0.9:
				rate[-1][1].append(total / num_4)
			else:
				rate[-1][1].append(0.)
		return rate

	# 让每一个epoch下，取不同的数据作为训练和验证
	def reget_data(self):
		test_indexs, val_indexs, train_indexs = self._split_data()
		setattr(self, 'test', test_indexs)
		setattr(self, 'val', val_indexs)
		setattr(self, 'train', train_indexs)
		rate = self.weight_calculate(train_indexs, val_indexs, self.total_data)
		setattr(self, 'rate', rate)








