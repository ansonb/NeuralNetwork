from node import Node
from ops import Relu
from ops import VariableOp
from ops import Mult
from ops import Add
from ops import Sub
from ops import PlaceholderOp
import numpy as np

np.random.seed(42)

class Sequential:
	def __init__(self):
		pass

	def __add_node(op, name, parents=[], variable_type='none', is_trainable=False):
		return Node(op,name,is_trainable=is_trainable,parents=parents,variable_type=variable_type)

	def add_layer(input_nodes, num_outputs, name=None):
		if name==None:
			name = str(random.random(1000000000,2000000000))
		layer_name = name

		#create output nodes going from inputs
		node_count = 0
		arr = [[None for _ in range(len(input_nodes))] for _ in range(num_outputs)]

		for node_num, node in enumerate(input_nodes):
			for i in range(num_outputs):
				node_name = '{}_node_w_{}'.format(layer_name,node_count)
				w_node = Sequential.__add_node(VariableOp,node_name,variable_type='weight',is_trainable=True)

				node_name = '{}_node_mult_{}'.format(layer_name,node_count)
				mult_node = Sequential.__add_node(Mult,node_name,parents=[w_node,node],is_trainable=False)

				
				arr[i][node_num] = mult_node

				node_count += 1


		out_arr_0 = []
		node_count_0 = 0
		for nodes in arr:
			node_count_1 = 0
			sum_node = nodes[0]
			for node in nodes[1:]:
				node_name = '{}_node_add_out_{}_{}'.format(layer_name,node_count_0,node_count_1)
				sum_node = Sequential.__add_node(Add,node_name,parents=[node,sum_node],is_trainable=False)

				node_count_1 += 1
				
			node_name = '{}_node_b_{}'.format(layer_name,node_count_0)
			bias_node = Sequential.__add_node(VariableOp,node_name,variable_type='bias',is_trainable=True)

			node_name = '{}_node_add_{}'.format(layer_name,node_count_0)
			sum_node = Sequential.__add_node(Add,node_name,parents=[sum_node,bias_node],is_trainable=False)

			out_arr_0.append(sum_node)

			node_count_0 += 1

		out_arr_non_linear = []
		for index, node in enumerate(out_arr_0):
			node_name = '{}_node_relu_{}'.format(layer_name,index)
			nl_node = Sequential.__add_node(Relu,node_name,parents=[node],is_trainable=False) 

			out_arr_non_linear.append(nl_node)

		return out_arr_non_linear


	def create_placeholders(num_nodes, name=None):
		if name==None:
			name = str(random.random(1000000000,2000000000))

		placeholder_arr = []
		for i in range(num_nodes):
			node_name = '{}_node_pl_{}'.format(name,i)
			pl_node = Sequential.__add_node(PlaceholderOp,node_name,is_trainable=False) 

			placeholder_arr.append(pl_node)

		return placeholder_arr

	# TODO: could add more loss
	def add_loss(output_layer, true_outputs, name=None):
		if name==None:
			name = str(random.random(1000000000,2000000000))

		assert len(output_layer)==len(true_outputs), "The number of nodes in output_layer and true_outputs must be same"

		arr = []
		for index in range(len(output_layer)):
			cur_out = output_layer[index]
			true_out = true_outputs[index]

			node_name = '{}_node_sub_{}'.format(name,index)
			diff_node = Sequential.__add_node(Sub,node_name,parents=[cur_out,true_out],is_trainable=False)

			node_name = '{}_node_square_{}'.format(name,index)
			square_node = Sequential.__add_node(Mult,node_name,parents=[diff_node,diff_node],is_trainable=False)
			 
			arr.append(square_node)

		sum_node = arr[0]
		for index, node in enumerate(arr[1:]):
			node_name = '{}_node_add_{}'.format(name,index)
			sum_node = Sequential.__add_node(Add,node_name,parents=[node,sum_node],is_trainable=False)
		# print(sum_node.name)
		return sum_node

	#dfs on start_nodes to get all weight nodes in graph
	def __get_weights(nodes, weights_arr):
		if len(nodes)==0:
			return

		for node in nodes:
			if node.op.variable_type == 'weight':
				weights_arr.append(node)
			Sequential.__get_weights(node.parents,weights_arr)

	def init_weights(graph, _range):
		weights_arr = []
		Sequential.__get_weights([graph['entry_node']],weights_arr)
		for weight_node in weights_arr:
			weight_node.op.val = (np.random.random())*_range

	#dfs on start_nodes to get all bias nodes in graph
	def __get_bias(nodes, bias_arr,count=0):
		if len(nodes)==0:
			return
		

		for node in nodes:
			if node.op.variable_type == 'bias':
				bias_arr.append(node)
			Sequential.__get_bias(node.parents,bias_arr,count=count+1)

	def init_bias(graph, _range):
		bias_arr = []
		Sequential.__get_bias([graph['entry_node']],bias_arr)

		for bias_node in bias_arr:
			bias_node.op.val = (np.random.random())*_range

	# TODO: could also add layer nodes to graph
	def get_layer_by_name(name, nodes, layer_arr):
		if len(nodes)==0:
			return

		for node in nodes:
			# TODO: hardcoded string node_relu
			if '{}_node_relu'.format(name) in node.name:
				layer_arr.append()
				break
			Sequential.get_layer_by_name(node.children,bias_arr)


