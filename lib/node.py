class Node:

	def __init__(self, op_class, name, children=[], parents=[], variable_type='none', is_trainable=False):
		self.op = op_class(parents,variable_type=variable_type,node_name=name)
		self.parents = parents
		self.children = children.copy()
		self.name = name

		self.is_trainable = is_trainable

		for parent in parents:
			parent.add_child(self)


	def node_present(self, node_name, node_array):
		for n in node_array:
			if node_name==n.name:
				return True

		return False

	def add_child(self, node):
		if not self.node_present(node.name,self.children):
			self.children.append(node)

	def get_consumers(self):
		return self.children

	def get_inputs(self):
		return self.parents

	def get_operation(self):
		return self.op

	def apply_grad(self, grad, learning_rate):
		if self.is_trainable==False:
			return

		# print('node name',self.name)
		# print('op.val before',self.op.val)
		# print('grad',grad)
		self.op.val = self.op.val - learning_rate*grad
		# print('op.val after',self.op.val)
		# print()