from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "forward must be called before backward."
        assert grad.shape[0] == self.input.shape[0]

        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride)
        if isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4
        assert X.shape[1] == self.in_channels
        self.input = X

        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        else:
            X_padded = X

        out_h = (X_padded.shape[2] - kernel_h) // stride_h + 1
        out_w = (X_padded.shape[3] - kernel_w) // stride_w + 1
        assert out_h > 0 and out_w > 0

        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded, (kernel_h, kernel_w), axis=(2, 3)
        )
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]
        output = np.einsum('nchwkl,ockl->nohw', windows, self.W)
        return output + self.b

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None, "forward must be called before backward."
        assert grads.ndim == 4

        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(self.input, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        else:
            X_padded = self.input

        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded, (kernel_h, kernel_w), axis=(2, 3)
        )
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]

        self.grads['W'] = np.einsum('nohw,nchwkl->ockl', grads, windows)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dX_padded = np.zeros_like(X_padded)
        out_h, out_w = grads.shape[2], grads.shape[3]
        for i in range(out_h):
            h_start = i * stride_h
            for j in range(out_w):
                w_start = j * stride_w
                dX_padded[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += np.einsum(
                    'no,ockl->nckl', grads[:, :, i, j], self.W
                )

        if pad_h > 0 or pad_w > 0:
            return dX_padded[:, :, pad_h:dX_padded.shape[2] - pad_h, pad_w:dX_padded.shape[3] - pad_w]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Flatten(Layer):
    """
    Flatten all non-batch dimensions before feeding convolutional features into linear layers.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        assert self.input_shape is not None, "forward must be called before backward."
        return grads.reshape(self.input_shape)

class Dropout(Layer):
    """
    Inverted dropout. During training it masks activations and rescales the remaining
    values; during evaluation it is an identity layer.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training or self.p == 0:
            self.mask = np.ones_like(X)
            return X
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, grads):
        assert self.mask is not None, "forward must be called before backward."
        return grads * self.mask

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.predicts = None
        self.labels = None
        self.probs = None
        self.one_hot_labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = np.asarray(labels)
        assert predicts.ndim == 2
        assert predicts.shape[0] == labels.shape[0]

        self.predicts = predicts
        self.labels = labels.astype(np.int64)
        batch_size, num_classes = predicts.shape

        if labels.ndim == 2:
            assert labels.shape == predicts.shape
            self.one_hot_labels = labels.astype(predicts.dtype)
            label_indices = np.argmax(labels, axis=1)
        else:
            assert np.all((self.labels >= 0) & (self.labels < num_classes))
            self.one_hot_labels = np.zeros((batch_size, num_classes), dtype=predicts.dtype)
            self.one_hot_labels[np.arange(batch_size), self.labels] = 1
            label_indices = self.labels

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        eps = 1e-12
        clipped_probs = np.clip(self.probs, eps, 1.0)
        loss = -np.mean(np.log(clipped_probs[np.arange(batch_size), label_indices]))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.one_hot_labels is not None
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            self.grads = (self.probs - self.one_hot_labels) / batch_size
        else:
            eps = 1e-12
            self.grads = -self.one_hot_labels / (np.clip(self.probs, eps, 1.0) * batch_size)
        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
