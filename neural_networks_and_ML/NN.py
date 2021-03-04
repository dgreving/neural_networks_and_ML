import numpy as np


class Operation:
    def __init__(self):
        self.input = None
        self.grad_input = None
        self.output = None
        self.grad_output = None

    def forward(self, input_):
        self.input = input_
        self.output = self.eval_output()
        return self.output

    def backward(self, grad_output):
        self.grad_output = grad_output
        self.grad_input = self.eval_input_gradient()
        return self.grad_input

    def eval_output(self):
        raise NotImplementedError

    def eval_input_gradient(self):
        raise NotImplementedError


class ParameterOperation(Operation):
    def __init__(self, parameter_matrix):
        super().__init__()
        self.parameter_matrix = parameter_matrix
        self.grad_parameters = None

    def backward(self, grad_output):
        self.grad_output = grad_output
        self.grad_input = self.eval_input_gradient()
        self.grad_parameters = self.eval_parameter_gradient()
        return self.grad_input

    def eval_parameter_gradient(self):
        raise NotImplementedError


class ApplyWeights(ParameterOperation):
    def __init__(self, weights):
        super().__init__(weights)  # adds attribute 'parameter_matrix' to instance

    def eval_output(self):
        return np.dot(self.input, self.parameter_matrix)

    def eval_input_gradient(self):
        return np.dot(self.grad_output, self.parameter_matrix.T)

    def eval_parameter_gradient(self):
        return np.dot(self.input.T, self.grad_output)


class ApplyBiases(ParameterOperation):
    def __init__(self, biases):
        super().__init__(biases)

    def eval_output(self):
        return self.input + self.parameter_matrix

    def eval_input_gradient(self):
        return self.grad_output

    def eval_parameter_gradient(self):
        return np.sum(self.grad_output, axis=0, keepdims=True)


class Linear(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return self.input

    def eval_input_gradient(self):
        return np.ones_like(self.grad_output)


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return 1. / (1 + np.exp(-self.input))

    def eval_input_gradient(self):
        return self.output * (1 - self.output) * self.grad_output


class Tanh(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return np.tanh(self.input)

    def eval_input_gradient(self):
        return 1 - self.output**2

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Layer:
    def __init__(self, n_neurons, weight_initialisation='glorot'):
        self.n_neurons = n_neurons
        self.weight_initialisation = weight_initialisation
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

    def forward(self, input_):
        self.input = input_
        if not self._input_known:
            self.set_up()
            self._input_known = True

        for op in self.ops:
            input_ = op.forward(input_)
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
    def __init__(self, n_neurons, activation_func, weight_initialisation='glorot'):
        super().__init__(n_neurons, weight_initialisation)
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


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Loss:
    def __init__(self):
        self.simulated = None
        self.true = None
        self.output = None
        self.input_gradient = None

    def forward(self, simulated, true):
        self.simulated = simulated
        self.true = true
        self.output = self.eval_output()
        return self.output

    def backward(self):
        self.input_gradient = self.eval_input_gradient()
        return self.input_gradient

    def eval_output(self):
        raise NotImplementedError

    def eval_input_gradient(self):
        raise NotImplementedError


class MeanSquaredLoss(Loss):
    def _init__(self):
        super().__init__()

    def eval_output(self):
        return np.sum((self.simulated - self.true)**2) / self.simulated.shape[0]

    def eval_input_gradient(self):
        return 2 * (self.simulated - self.true) / self.simulated.shape[0]


class SoftMaxCrossEntropyLoss(Loss):
    def __init__(self, softmax_clip_eps=None):
        super().__init__()
        self.softmax_clip_eps = softmax_clip_eps

    def softmax(self):
        a = np.exp(self.simulated - np.max(self.simulated, axis=1, keepdims=True))
        rv = a / np.sum(a, axis=1, keepdims=True)
        return rv

    def eval_output(self):
        sm = self.softmax()
        if self.softmax_clip_eps is not None:
            sm = np.clip(sm, self.softmax_clip_eps, 1 - self.softmax_clip_eps)
        val = -self.true * np.log(sm) - (1 - self.true) * np.log(1 - sm)
        return np.sum(val) / self.simulated.shape[0]

    def eval_input_gradient(self):
        return (self.softmax() - self.true) / self.simulated.shape[0]


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class NN:
    def __init__(self, loss_func=MeanSquaredLoss()):
        self.layers = []
        self.loss_func = loss_func
        self.output = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_):
        for layer in self.layers:
            input_ = layer.forward(input_)
        self.output = input_
        return self.output

    def backward(self, loss_gradient):
        output_gradient = loss_gradient
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def eval(self, x, y):
        simulated = self.forward(x)
        loss = self.loss_func.forward(simulated, true=y)
        loss_gradient = self.loss_func.backward()
        self.backward(loss_gradient)
        return loss

    @property
    def parameters(self):
        for layer in self.layers:
            for parameter_matrix in layer.parameters:
                yield parameter_matrix

    @property
    def parameter_gradients(self):
        for layer in self.layers:
            for parameter_gradient in layer.parameter_gradients:
                yield parameter_gradient


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.nn = None

    def add_neural_net(self, nn):
        self.nn = nn

    def optimization_step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def optimization_step(self):
        for parameter, parameter_grad in zip(self.nn.parameters, self.nn.parameter_gradients):
            parameter -= self.learning_rate * parameter_grad


class MomentumSGD(Optimizer):
    """Stochastic Gradient Descent including momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.8):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = []
        self._input_known = False

    def optimization_step(self):
        if not self._input_known:
            self.velocities = [np.zeros_like(gradient) for gradient in self.nn.parameter_gradients]
            self._input_known = True
        for v, parameter, parameter_grad in zip(self.velocities, self.nn.parameters, self.nn.parameter_gradients):
            v = self.update_velocity(v, parameter_grad)
            parameter -= self.learning_rate * v

    def update_velocity(self, current_velocity, parameter_grad):
        current_velocity *= self.momentum
        current_velocity += parameter_grad
        return current_velocity


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Trainer:
    def __init__(self, neural_net: NN, optimizer: Optimizer):
        self.nn = neural_net
        self.optimizer = optimizer
        self.optimizer.add_neural_net(self.nn)

    def optimize(self, x_train, y_train, epochs=1, batch_size=100):
        from utils import create_batches
        for epoch in range(epochs):
            for X, y in create_batches(x_train, y_train, batch_size):
                loss = self.nn.eval(X, y)
                self.optimizer.optimization_step()
                print(loss)

