from abc import ABC, abstractmethod

class Op(ABC):
	def __init__(self, node_name, is_trainable=False):
		super().__init__()
		self.val = None
		self.variable_type = 'none'
		self.node_name = node_name
		

	def bprop(self, input_nodes, cur_node, target_node, grad_table):
		assert grad_table[cur_node.name]!=None, "grad_table[cur_node] should not be None"
		# grad_sum = 0
		# for node in input_nodes:
		# 	grad_sum += self.derivative(target_node) #diff wrt target_node
		# grad = grad_table[cur_node.name]*grad_sum
		grad = grad_table[cur_node.name]*self.derivative(target_node)
		# if self.node_name=='layer2_node_relu_0':
		# 	print('grad',grad,grad_table[cur_node.name],target_node.name,self.derivative(target_node))
		return grad

	def f(self, *args, **kwargs):
		pass

	@abstractmethod
	def default_derivative(self, *args, **kwargs):
		pass

	@abstractmethod
	def derivative(self, *args, **kwargs):
		pass

# TODO: could add more non linearity
class Relu(Op):
	def __init__(self, *args, **kwargs):
		assert len(args)==1, 'Length of inputs must be 1 to Relu op'
		super().__init__(kwargs['node_name'])
		self.inputs = args
		self.input_node_1 = args[0][0]
		self.derivative_dict = {}
		self.derivative_dict[self.input_node_1.name] = self.derivative_node_1
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1

	def f(self, *args, **kwargs):
		self.val = max(self.input_node_1.op.val,0)
		# self.val = self.input_node_1.op.val
		return self.val

	def derivative_node_1(self, *args, **kwargs):
		x = args[0].op.val
		return 0 if x<0 else 1
		# return x

	def default_derivative(self, *args, **kwargs):
		return self.input_node_1.op.derivative(args[0])

	def derivative(self, *args, **kwargs):
		if args[0].name == self.node_name:
			return 1

		return self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)


class Mult(Op):
	#operation is input_node_1*input_node_2
	def __init__(self, *args, **kwargs):
		assert len(args[0])==2, 'Length of inputs must be 2 to Mult op'
		super().__init__(kwargs['node_name'],is_trainable=False)
		self.inputs = args[0]
		self.input_node_1 = self.inputs[0]
		self.input_node_2 = self.inputs[1]
		self.derivative_dict = {}
		self.derivative_dict[self.input_node_1.name] = self.derivative_node_1
		self.derivative_dict[self.input_node_2.name] = self.derivative_node_2
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1
		
	# TODO: correct
	def f(self, *args, **kwargs):
		self.val = self.input_node_1.op.val * self.input_node_2.op.val
		# if self.node_name=='loss_node_square_0':
		# 	print(self.input_node_1.op.val,self.input_node_2.op.val,self.val)
		return self.val

	def derivative_node_1(self, *args, **kwargs):
		return  input_node_2.op.val

	def derivative_node_2(self, *args, **kwargs):
		return input_node_1.op.val

	def default_derivative(self, *args, **kwargs):
		return self.input_node_1.op.val*self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_2.op.val*self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs)

	# def derivative(self, *args, **kwargs):
	# 	return self.input_node_1.op.val*self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs) + self.input_node_2.op.val*self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)
	def derivative(self, *args, **kwargs):
		if args[0].name == self.node_name:
			return 1

		return self.input_node_1.op.val*self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_2.op.val*self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs) 

class Add(Op):
	#operation is input_node_1+input_node_2
	def __init__(self, *args, **kwargs):
		assert len(args[0])==2, 'Length of inputs must be 2 to Add op'
		super().__init__(kwargs['node_name'],is_trainable=False)
		self.inputs = args[0]
		self.input_node_1 = self.inputs[0]
		self.input_node_2 = self.inputs[1]
		self.derivative_dict = {}
		self.derivative_dict[self.input_node_1.name] = self.derivative_node_1
		self.derivative_dict[self.input_node_2.name] = self.derivative_node_2
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1
		

	def f(self, *args, **kwargs):
		self.val = self.input_node_1.op.val + self.input_node_2.op.val
		return self.val

	def derivative_node_1(self, *args, **kwargs):
		return  1

	def derivative_node_2(self, *args, **kwargs):
		return 1

	def default_derivative(self, *args, **kwargs):
		return self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs)


	# def derivative(self, *args, **kwargs):
	# 	return self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs) + self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)
	def derivative(self, *args, **kwargs):
		# print('derivative sub---->')
		# print(args[0].name,self.node_name)
		# if args[0].name=='loss_node_square_0' and self.node_name=='loss_node_add_0':
		# 	print('=============>')
		# 	print(self.input_node_2.op.derivative_dict.get(args[0].name)(*args,**kwargs) + self.input_node_1.op.derivative_dict.get(args[0].name)(*args,**kwargs))
		# 	print(self.input_node_2.name)
		# 	print(self.input_node_1.name)
		# 	print(args[0].name)
		if args[0].name == self.node_name:
			return 1

		return self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs)

class Sub(Op):
	#operation is input_node_1-input_node_2
	def __init__(self, *args, **kwargs):
		assert len(args[0])==2, 'Length of inputs must be 2 to Sub op'
		super().__init__(kwargs['node_name'],is_trainable=False)
		self.inputs = args[0]
		self.input_node_1 = self.inputs[0]
		self.input_node_2 = self.inputs[1]
		self.derivative_dict = {}
		self.derivative_dict[self.input_node_1.name] = self.derivative_node_1
		self.derivative_dict[self.input_node_2.name] = self.derivative_node_2
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1
		

	def f(self, *args, **kwargs):
		self.val = self.input_node_1.op.val - self.input_node_2.op.val
		return self.val

	def derivative_node_1(self, *args, **kwargs):
		return  1

	def derivative_node_2(self, *args, **kwargs):
		return -1

	def default_derivative(self, *args, **kwargs):
		return -1*self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs)

	# def derivative(self, *args, **kwargs):
	# 	return self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs) - self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)
	def derivative(self, *args, **kwargs):
		# print('derivative sub---->')
		# print(args[0].name,self.node_name)
		if args[0].name == self.node_name:
			return 1

		return -1*self.input_node_2.op.derivative_dict.get(args[0].name,self.input_node_2.op.default_derivative)(*args,**kwargs) + self.input_node_1.op.derivative_dict.get(args[0].name,self.input_node_1.op.default_derivative)(*args,**kwargs)

class VariableOp(Op):
	def __init__(self, *args, **kwargs):
		assert len(args)==1, 'Length of inputs must be 1 to VariableOp op'
		super().__init__(kwargs['node_name'],is_trainable=True)
		self.inputs = args
		self.derivative_dict = {}
		self.derivative_dict[kwargs['node_name']] = self.derivative_self
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1

		self.variable_type = kwargs['variable_type']
		

	def derivative_self(self, *args, **kwargs):
		return  1

	def default_derivative(self, *args, **kwargs):
		return 0

	def derivative(self, *args, **kwargs):
		return self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)



class PlaceholderOp(Op):
	def __init__(self, *args, **kwargs):
		assert len(args)==1, 'Length of inputs must be 1 to PlaceholderOp'
		super().__init__(kwargs['node_name'],is_trainable=False)
		self.inputs = args
		self.derivative_dict = {}
		self.derivative_dict[kwargs['node_name']] = self.derivative_self
		self.derivative_dict[self.node_name] = lambda *args, **kwargs: 1
		self.val = None
		
	def set_val(self, val):
		self.val = val

	def derivative_self(self, *args, **kwargs):
		return  1

	def default_derivative(self, *args, **kwargs):
		return 0

	def derivative(self, *args, **kwargs):
		return self.derivative_dict.get(args[0].name,self.default_derivative)(*args,**kwargs)