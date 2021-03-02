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


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return 1. / (1 + np.exp(-self.input))

    def eval_input_gradient(self):
        return self.output * (1 - self.output) * self.grad_output


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
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
    def __init__(self, n_neurons, activation_func=Sigmoid()):
        super().__init__(n_neurons)
        self.activation_func = activation_func

    def set_up(self):
        weights = np.random.randn(self.input.shape[1], self.n_neurons)
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
        self.simulated = None
        self.true = None

    def eval_output(self):
        return np.sum((self.simulated - self.true)**2) / self.simulated.shape[0]

    def eval_input_gradient(self):
        return 2 * (self.simulated - self.true)


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
