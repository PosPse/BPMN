import sys, os
import torch
import random
import re
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		#这里还是考虑加一层
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds, data_center, nodes_id):
		tokens = getattr(data_center, 'total_data')[nodes_id[0]][1]
		activity_num = 0
		activity_index = []
		for i in range(0, len(tokens)):
			if tokens[i] == '[activity]':
				activity_num += 1
				activity_index.append(i)
		result = self.concat(embeds, activity_num, activity_index)
		#logists = self.layer(result)
		#logists = torch.softmax(self.layer(result), 1)
		logists = torch.log_softmax(self.layer(result), 1)
		return logists

	def concat(self, x, activity_num, activity_index):
		# 解决左边的矩阵
		first_activity_index = activity_index[0]
		n = activity_num - 1
		result_left = x[first_activity_index].clone().reshape(1, -1)
		result_left = result_left.repeat(n, 1)

		n -= 1
		for j in range(1, len(activity_index)-1):
			temp = x[activity_index[j]].clone().reshape(1, -1)
			temp = temp.repeat(n, 1)
			result_left = torch.cat((result_left, temp), 0)
			n = n - 1
		# 准备右矩阵
		n = activity_num
		result_right = x[activity_index[1]].clone().reshape(1, -1)
		current_start_id = 2
		for j in range(2, len(activity_index)):
			temp = x[activity_index[j]].clone().reshape(1, -1)
			result_right = torch.cat((result_right, temp), 0)
		while current_start_id != n:
			for j in range(current_start_id, len(activity_index)):
				temp = x[activity_index[j]].clone().reshape(1, -1)
				result_right = torch.cat((result_right, temp), 0)
			current_start_id += 1

		# 左右拼接
		if len(result_left) != len(result_right):
			print("左右矩阵长度不同，算法错误")
		result = torch.cat((result_left, result_right), 1)
		return result

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, n, gcn=False):
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size

		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else n * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		combined = F.tanh(self.weight.mm(combined.t())).t()
		return combined

class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, agg_function_model, data_center, agg_func, gcn=False):
		super(GraphSage, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.gcn = gcn
		self.agg_func = agg_func

		self.raw_features = raw_features
		self.adj_lists = adj_lists
		self.data_center = data_center
		self.agg_function_model = agg_function_model

		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			n = 0
			if self.agg_func == 'BiLSTM':
				n = 3
			else:
				n = 2
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, n = n, gcn=self.gcn))

	def forward(self, nodes_id):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		cur_adj = self.adj_lists[nodes_id[0]]
		nodes_batch = cur_adj.keys()
		lower_layer_nodes = list(nodes_batch)
		nodes_batch_layers = [(lower_layer_nodes,)]
		lower_samp_neighs = self._get_unique_neighs_list(nodes_id, lower_layer_nodes)
		for i in range(self.num_layers):
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs))

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features[nodes_id[0]].clone()
		for index in range(1, self.num_layers+1):
			nb = nodes_batch_layers[index][0]
			pre_neighs = nodes_batch_layers[index-1]
			aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
			sage_layer = getattr(self, 'sage_layer'+str(index))
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)

			pre_hidden_embs[list(nodes_batch)] = cur_hidden_embs
		return pre_hidden_embs

	def _get_unique_neighs_list(self,nodes_id, nodes):
		_set = set
		to_neighs = [self.adj_lists[nodes_id[0]][int(node)] for node in nodes]
		samp_neighs = to_neighs
		#混入自己,自己是自己的邻居
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		return samp_neighs

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
		unique_nodes_list, samp_neighs = pre_neighs
		unique_nodes = [i for i in range(0,pre_hidden_embs.shape[0])]
		assert len(nodes) == len(samp_neighs)
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		assert (False not in indicator)
		#samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
		embed_matrix = pre_hidden_embs
		mask = torch.zeros(len(samp_neighs), embed_matrix.shape[0])
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1

		if self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True)
			for i in range(0, len(num_neigh)):
				if num_neigh[i][0] == 0:
					num_neigh[i][0] = 1
			mask = mask.div(num_neigh)
			aggregate_feats = mask.mm(embed_matrix)
			return aggregate_feats

		elif self.agg_func == 'MAX':
			indexs = [x.nonzero() for x in mask==1]
			aggregate_feats = []
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)
			return aggregate_feats

		elif self.agg_func == 'LSTM':
			mask = mask.int()
			mask = mask.tolist()
			aggregate_feats = torch.tensor([float(i) for i in range(0, 128)]).reshape(1, -1)
			for node in mask:
				node_batch = torch.tensor([float(i) for i in range(0, 128)]).reshape(1, -1)
				for i in range(0, len(node)):
					if node[i] == 1:
						node_batch = torch.cat([node_batch, embed_matrix[i].reshape(1, -1)], 0)
				result, (h_n, c_n) = self.agg_function_model(node_batch[1:].unsqueeze(0))
				aggregate_feat = h_n.squeeze(0)
				aggregate_feats = torch.cat([aggregate_feats, aggregate_feat], 0)
			return aggregate_feats[1:]

		elif self.agg_func == 'BiLSTM':
			node_id = 0 # 记录当前在处理的结点的编号，确保语义融合的结果拼接到对应的结点上
			mask = mask.int()
			mask = mask.tolist()
			aggregate_feats = torch.tensor([float(i) for i in range(0, 256)]).reshape(1, -1)
			for node in mask:
				node_batch = torch.tensor([float(i) for i in range(0, 128)]).reshape(1, -1)
				for i in range(0, len(node)):
					if node[i] == 1:
						node_batch = torch.cat([node_batch, embed_matrix[i].reshape(1, -1)], 0)
				result, (h_n, c_n) = self.agg_function_model(node_batch[1:].unsqueeze(0))
				aggregate_feat = result.squeeze(0)[node_id].reshape(1, -1)
				aggregate_feats = torch.cat([aggregate_feats, aggregate_feat], 0)
				node_id += 1
			return aggregate_feats[1:]

		elif self.agg_func == 'Transformer':
			mask = mask.int().tolist()
			aggregate_feats = torch.tensor([float(i) for i in range(0, 128)]).reshape(1, -1)
			for i in range(0, len(mask)):
				# 偏移值，一串结点做语义融合，但是究竟是为哪个结点做融合？
				move = 0
				node_batch = torch.tensor([float(i) for i in range(0, 128)]).reshape(1, -1)
				for j in range(0, len(mask[i])):
					if mask[i][j] == 1:
						if(j < i):
							move += 1
						node_batch = torch.cat([node_batch, embed_matrix[i].reshape(1, -1)], 0)
				# 我们期望transformer的输出和它的输入一样,transformer出问题了，输入和输出的维度不一样，原因是把padding算进去了
				cur_tensor_num = node_batch.shape[0] - 1
				result = self.agg_function_model(node_batch[1:].unsqueeze(0)).squeeze(0)[0:cur_tensor_num]
				aggregate_feat = result.squeeze(0)[move].reshape(1, -1)
				aggregate_feats = torch.cat([aggregate_feats, aggregate_feat], 0)
			return aggregate_feats[1:]