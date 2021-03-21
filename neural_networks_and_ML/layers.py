import numpy as np
from parameterized_operations import ApplyWeights, ApplyBiases, Conv2D
from parameterless_operations import Dropout, Pool, Flatten


class Layer:
    def __init__(self, n_neurons, weight_initialisation='glorot', dropout=1):
        self.n_neurons = n_neurons
        self.weight_initialisation = weight_initialisation
        self.dropout = dropout
        self.input = None
        self.grad_input = None
        self.output = None
        self.grad_output = None
        self.params = []
        self.ops = []
        self.para_grads = []

        self._input_known = False

    def set_up(self):
        raise NotImplementedError

    def forward(self, input_, training_run=False):
        self.input = input_
        if not self._input_known:
            self.set_up()
            self._input_known = True

        for op in self.ops:
            input_ = op.forward(input_, training_run=training_run)
        self.output = input_
        return self.output

    def backward(self, grad_output):
        self.grad_output = grad_output
        for op in reversed(self.ops):
            grad_output = op.backward(grad_output)
        self.grad_input = grad_output
        return self.grad_input

    @property
    def parameters(self):
        for op in self.ops:
            try:
                yield op.parameter_matrix
            except AttributeError:
                pass

    @property
    def parameter_gradients(self):
        for op in self.ops:
            try:
                yield op.grad_parameters
            except AttributeError:
                pass


class DenseLayer(Layer):
    def __init__(self, n_neurons, activation_func, weight_initialisation='glorot', dropout=1.):
        super().__init__(n_neurons, weight_initialisation, dropout)
        self.activation_func = activation_func

    def set_up(self):
        if self.weight_initialisation == 'glorot':
            scale = 2. / (self.input.shape[1] + self.n_neurons)
        else:
            scale = 1.
        weights = np.random.randn(self.input.shape[1], self.n_neurons) * scale
        biases = np.random.randn(1, self.n_neurons)
        self.ops.append(ApplyWeights(weights))
        self.ops.append(ApplyBiases(biases))
        self.ops.append(self.activation_func)
        if self.dropout < 1.:
            self.ops.append(Dropout(keep_probability=self.dropout))


class ConvLayer(Layer):
    def __init__(self, n_channels, kernel_size, activation_func, weight_init='glorot', dropout=1, flatten=True):
        super().__init__(n_neurons=0, dropout=dropout, weight_initialisation=weight_init)
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.input = None
        self.output = None
        self.activation_func = activation_func

        self.flatten = flatten
        self._input_known = False

    def set_up(self):
        num_input_channels, num_output_channels = self.input.shape[1], self.n_channels
        if self.weight_initialisation == 'glorot':
            scale = 2. / (num_input_channels + num_output_channels)
        else:
            scale = 1.

        weights = np.random.normal(
            scale=scale,
            size=(num_input_channels, num_output_channels, self.kernel_size, self.kernel_size),
        )
        self.ops.append(Conv2D(weights))
        self.ops.append(self.activation_func)
        if self.flatten:
            self.ops.append(Flatten())
        if self.dropout < 1.:
            self.ops.append(Dropout(keep_probability=self.dropout))


class PoolingLayer(Layer):
    def __init__(self, pool_size=2, flatten=False):
        super().__init__(n_neurons=0)
        self.pool_size = pool_size
        self.flatten = flatten

    def set_up(self):
        self.ops.append(Pool(pool_size=self.pool_size))
        if self.flatten:
            self.ops.append(Flatten())
