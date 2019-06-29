from utils import CustomEncoder
import json
from node import Node
import ops
import numpy as np

class Graph:
	def __init__(self, entry_node):
		self.graph = {}
		self.learning_rate = 0.01
		self.__create_graph([entry_node],self.graph)
		

	def build_grad(self, V, G, grad_table):
		grad = 0
		
		consumers = V.get_consumers()
		if len(consumers)==0:
			grad_table[V.name] = 1
			return

		for C in consumers:
			op = C.get_operation()
			self.build_grad(C,G,grad_table)
			grad += op.bprop(C.get_inputs(),C,V,grad_table)

		grad_table[V.name] = grad

	# target nodes will be all trainable nodes
	def __find_gradients(self, target_nodes, output_node, G):
		grad_table = self.init_grad_table(target_nodes,output_node,G)
		# print(target_nodes)
		for node in target_nodes:
			# print(node.name)
			self.build_grad(node, G, grad_table)
			# print(grad_table[node.name])
			# print()

		return grad_table

	def __apply_gradients(self, grad_table, G):
		for node_name, grad in grad_table.items():
			if grad is not None:
				G[node_name].apply_grad(grad,self.learning_rate)

	def init_grad_table(self, target_nodes, output_node, G):
		grad_table = {}
		for node_name, _ in G.items():
			if node_name == 'entry_node':
				grad_table[G[node_name].name] = 1
				continue
			grad_table[node_name] = None

		return grad_table

	#dfs on start nodes to get all nodes in graph
	def __create_graph(self, nodes, graph, level=0):
		
		if level==0:
			graph['entry_node'] = nodes[0]
		if nodes==None:
			return

		for node in nodes:
			graph[node.name] = node
			self.__create_graph(node.parents,graph,level=level+1)


	def save_graph(self, graph, path):
		graph_json = {}
		graph_json['metadata']['learning_rate'] = self.learning_rate
		
		graph_json['entry_node'] = []
		graph_json['entry_node'] = graph['entry_node'].name

		nodes_to_ignore = ['entry_node']

		for node_name, node in graph:
			if node_name in nodes_to_ignore:
				continue
			graph_json[node.name] = {}
			graph_json[node.name]['op_name'] = node.op.__class__.__name__
			graph_json[node.name]['op_val'] = node.op.val
			graph_json[node.name]['op_inputs'] = [i.name for i in node.op.inputs]
			graph_json[node.name]['variable_type'] = node.op.variable_type
			graph_json[node.name]['parents'] = [i.name for i in node.parents]
			graph_json[node.name]['children'] = [i.name for i in node.children]
			graph_json[node.name]['is_trainable'] = node.is_trainable

		with open(path, 'w') as f:
			json.dump(graph_json,f,indent=4,cls=CustomEncoder)

	def add_node(self, op, name, parents, variable_type='none', is_trainable=True):
		return Node(op,name,is_trainable,parents=parents,variable_type=variable_type)

	def restore_node(self, graph_json, graph, node_name):		
		if graph.get(node_name,None) != None:
			return graph[node_name]

		parent_nodes = graph_json[node_name]['parents']
		parents= []
		for parent_node_name in parent_nodes:
			if graph.get(parent_node_name,None)==None:
				parents.append(self.restore_nodes(graph_json,graph,parent_node_name))
			else:
				parents.append(graph[parent_node_name])

		node_data = graph_json[node_name]
		op_class = getattr(ops,node_data['op_name'])
		node = self.add_node(op_class,node_name,parents,variable_type=node_data['variable_type'],is_trainable=node_data['is_trainable']) 
		graph[node_name] = node

		return node

	def restore_graph(self, path_to_restore_from):
		with open(path_to_restore_from, 'r') as f:
			graph_json = json.load(f)

		graph = {}
		graph['entry_point'] = []
		for node_name in graph_json['entry_nodes']:
			graph['entry_point'].append(self.restore_node(graph_json,graph,node_name))

		return graph

	def __init_placeholders(self):
		for node_name, node in self.graph.items():
			if node_name=='entry_nodes':
				continue

			if node.op.__class__.__name__ == 'PlaceholderOp':
				node.op.set_val(None)
		

	def __all_placeholders_initialised(self):
		for node_name, node in self.graph.items():
			if node_name=='entry_nodes':
				continue

			if node.op.__class__.__name__ == 'PlaceholderOp':
				if node.op.val == None:
					return False

		return True
		

	def __f_compute(self, node):
		for parent in node.parents:
			self.__f_compute(parent)

		node.op.f()

	def __forward_step(self, feed_dict):
		self.__init_placeholders()
		for node_name, value in feed_dict.items():
			self.graph[node_name].op.val = value
			# print(node_name)
			# print(self.graph[node_name].op.val)
			# print()
		assert self.__all_placeholders_initialised(), 'All placeholders not initialised'

		# for node in self.graph['entry_nodes']:
		# 	self.__f_compute(node)
		self.__f_compute(self.graph['entry_node'])

	def __get_trainable_nodes(self):
		nodes = []
		for node_name, node in self.graph.items():
			if node_name == 'entry_nodes':
				continue

			if node.is_trainable:
				nodes.append(node)

		return nodes

	def __backward_step(self):
		target_nodes = self.__get_trainable_nodes()
		grad_table = self.__find_gradients(target_nodes, self.graph['entry_node'],self.graph)
		# print('grad_table',grad_table)
		self.__apply_gradients(grad_table,self.graph)

	def train_step(self, feed_dict):
		self.__forward_step(feed_dict)
		self.__backward_step()

		return self.graph['entry_node'].op.val

	def test_step(self, feed_dict):
		self.__forward_step(feed_dict)