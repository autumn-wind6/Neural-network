from .op import *
import pickle
import numpy as np

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list
        self.dropout_list = dropout_list

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout_list is not None and dropout_list[i] > 0:
                        self.layers.append(Dropout(dropout_list[i]))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.dropout_list = None

        if len(param_list) > 2 and isinstance(param_list[2], dict) and 'dropout_list' in param_list[2]:
            self.dropout_list = param_list[2]['dropout_list']
            param_offset = 3
        else:
            param_offset = 2

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + param_offset]['W']
            layer.b = param_list[i + param_offset]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + param_offset]['weight_decay']
            layer.weight_decay_lambda = param_list[i + param_offset]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
                if self.dropout_list is not None and self.dropout_list[i] > 0:
                    self.layers.append(Dropout(self.dropout_list[i]))
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, {'dropout_list': self.dropout_list}]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, in_channels=1, input_size=28, num_classes=10, conv_channels=8, kernel_size=3, stride=1, padding=1, lambda_list=None, dropout_rate=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lambda_list = lambda_list
        self.dropout_rate = dropout_rate
        self.optimizable = False

        conv_std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        conv_layer = conv2D(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            initialize_method=lambda size: np.random.normal(0, conv_std, size=size),
        )
        conv_layer.b = np.zeros_like(conv_layer.b)
        conv_layer.params['b'] = conv_layer.b

        conv_out_size = (input_size + 2 * padding - kernel_size) // stride + 1
        flatten_dim = conv_channels * conv_out_size * conv_out_size
        linear_std = np.sqrt(2.0 / flatten_dim)
        linear_layer = Linear(
            in_dim=flatten_dim,
            out_dim=num_classes,
            initialize_method=lambda size: np.random.normal(0, linear_std, size=size),
        )
        linear_layer.b = np.zeros_like(linear_layer.b)
        linear_layer.params['b'] = linear_layer.b

        if lambda_list is not None:
            conv_layer.weight_decay = True
            conv_layer.weight_decay_lambda = lambda_list[0]
            linear_layer.weight_decay = True
            linear_layer.weight_decay_lambda = lambda_list[1]

        self.layers = [conv_layer, ReLU(), Flatten()]
        if dropout_rate > 0:
            self.layers.append(Dropout(dropout_rate))
        self.layers.append(linear_layer)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = self._prepare_input(X)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            state = pickle.load(f)

        config = state['config']
        self.__init__(**config)
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved_param in zip(optimizable_layers, state['params']):
            layer.W = saved_param['W']
            layer.b = saved_param['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved_param['weight_decay']
            layer.weight_decay_lambda = saved_param['lambda']
        
    def save_model(self, save_path):
        state = {
            'model': 'Model_CNN',
            'config': {
                'in_channels': self.in_channels,
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'conv_channels': self.conv_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': self.padding,
                'lambda_list': self.lambda_list,
                'dropout_rate': self.dropout_rate,
            },
            'params': [],
        }
        for layer in self.layers:
            if layer.optimizable:
                state['params'].append({
                    'W' : layer.params['W'],
                    'b' : layer.params['b'],
                    'weight_decay' : layer.weight_decay,
                    'lambda' : layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

    def _prepare_input(self, X):
        if X.ndim == 2:
            expected_dim = self.in_channels * self.input_size * self.input_size
            assert X.shape[1] == expected_dim
            return X.reshape(X.shape[0], self.in_channels, self.input_size, self.input_size)
        if X.ndim == 3:
            assert self.in_channels == 1
            assert X.shape[1] == self.input_size and X.shape[2] == self.input_size
            return X.reshape(X.shape[0], self.in_channels, self.input_size, self.input_size)
        assert X.ndim == 4
        assert X.shape[1] == self.in_channels
        return X

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
