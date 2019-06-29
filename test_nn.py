import sys
sys.path.append('./lib')

from lib.nn import Graph
from lib.sequential import Sequential
import numpy as np

num_inputs = 2
num_outputs = 1
num_units_1 = 2
num_units_2 = num_outputs

plx = Sequential.create_placeholders(num_inputs,name='plx')
ply = Sequential.create_placeholders(num_outputs,name='ply')
l1 = Sequential.add_layer(plx,num_units_1,name='layer1')
l2 = Sequential.add_layer(l1,num_units_2,name='layer2')
loss = Sequential.add_loss(l2,ply,name='loss')

graph = Graph(loss)
Sequential.init_bias(graph.graph,0)
Sequential.init_weights(graph.graph,1)

np.random.seed(42)
data_x = [[1,0],[0,1],[0.8,0.1]]
data_y = [1,0,1]
epochs = 200
for epoch_num in range(epochs):
	for index in range(len(data_x)):
		x = data_x[index]
		y = data_y[index]

		feed_dict = {}
		for index_1, node in enumerate(plx):
			feed_dict[plx[index_1].name] = x[index_1]
		feed_dict[ply[0].name] = y

		loss_val = graph.train_step(feed_dict)
		print('Loss: {}'.format(loss_val))
	print()
		# print(graph.graph['layer1_node_w_0'].op.val)
