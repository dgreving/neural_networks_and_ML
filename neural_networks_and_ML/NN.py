import numpy as np
from scipy.ndimage import convolve


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


class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return np.reshape(self.input, (self.input.shape[0], -1))

    def eval_input_gradient(self):
        return np.reshape(self.grad_output, self.input.shape)


class Pool(Operation):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def eval_output(self):
        n_batch, n_channels = self.input.shape[0], self.input.shape[1]
        orig_size_x, orig_size_y = self.input.shape[-2], self.input.shape[-1]
        new_size_x, new_size_y = np.ceil(orig_size_x / self.pool_size), np.ceil(orig_size_y / self.pool_size)

        input_grad = np.zeros_like(self.input)
        output = np.empty((n_batch, n_channels, int(new_size_x), int(new_size_y)))

        for x in range(0, self.input.shape[2], self.pool_size):
            for y in range(0, self.input.shape[3], self.pool_size):
                window = self.input[..., x:x + self.pool_size, y:y + self.pool_size]
                flat_window = np.reshape(window, (-1, np.prod(window.shape[-2:])))
                flat_output_grad = np.reshape(input_grad, (-1, orig_size_x, orig_size_y))
                depth = flat_window.shape[0]

                flat_output = np.reshape(output, (-1, int(new_size_x), int(new_size_y)))
                flat_output[np.arange(depth), x // self.pool_size, y // self.pool_size] = np.max(flat_window, axis=-1)

                idx = np.argmax(flat_window, axis=-1)
                new_index = np.unravel_index(idx, window.shape)
                flat_output_grad[np.arange(depth), x + new_index[2], y + new_index[3]] = 1
        self.output = output
        self.grad_input = input_grad
        return output

    def eval_input_gradient(self):
        return self.grad_input


class ConvChannel(ParameterOperation):
    """Kind of the pendant to a single neuron containing a single operation in the case of a DenseLayer...(?)"""

    def __init__(self, weight_kernel):
        """
        weight_kernel has to be of shape (M, N, x_px, y_px), with M being the number of input channels and
        N being the number of output channels, i.e. every channel of the input gets convoluted with a separate
        kernel to produce exactly one output channel. x_px and y_px are the lateral dimensions of each channel.
        """
        super().__init__(weight_kernel)
        self.kernel_spread = 3

    def spread_kernel(self, kernel):
        wide_kernel = np.zeros((kernel.shape[:1] + (2*self.kernel_spread+1, 2*self.kernel_spread+1)))
        wide_kernel[..., 0, 0] = kernel[..., 0, 0]  # top left
        wide_kernel[..., self.kernel_spread, 0] = kernel[..., 1, 0]  # top centre
        wide_kernel[..., 2*self.kernel_spread, 0] = kernel[..., 2, 0]  # top right
        wide_kernel[..., 0, self.kernel_spread] = kernel[..., 0, 1]  # centre left
        wide_kernel[..., self.kernel_spread, self.kernel_spread] = kernel[..., 1, 1]  # centre
        wide_kernel[..., 2*self.kernel_spread, self.kernel_spread] = kernel[..., 2, 1]  # centre right
        wide_kernel[..., 0, 2*self.kernel_spread] = kernel[..., 0, 2]  # bottom left
        wide_kernel[..., self.kernel_spread, 2*self.kernel_spread] = kernel[..., 1, 2]  # bottom centre
        wide_kernel[..., 2*self.kernel_spread, 2*self.kernel_spread] = kernel[..., 2, 2]  # bottom right
        return wide_kernel


    def eval_output(self):
        batch_size = self.input.shape[0]
        n_input_channels = self.input.shape[1]
        n_output_channels = self.parameter_matrix.shape[1]  # para.shape = (input_channels, output_channels, x, y)

        self.output = np.zeros((batch_size, n_output_channels, self.input.shape[-2], self.input.shape[-1]))
        for out_idx in range(n_output_channels):
            for in_idx in range(n_input_channels):
                kernel = self.parameter_matrix[np.newaxis, in_idx, out_idx, :, :]
                kernel = self.spread_kernel(kernel)
                self.output[:, out_idx, :, :] += convolve(self.input[:, in_idx, :, :], kernel)
        return self.output
        # return convolve(self.input, self.parameter_matrix)

    def eval_input_gradient(self):
        n_input_channels = self.input.shape[1]
        n_output_channels = self.parameter_matrix.shape[1]  # para.shape = (input_channels, output_channels, x, y)

        self.grad_input = np.zeros_like(self.input)

        for out_idx in range(n_output_channels):
            for in_idx in range(n_input_channels):
                kernel = self.parameter_matrix[in_idx, out_idx, :, :]
                inv_kernel = np.flipud(np.fliplr(kernel))[np.newaxis, :, :]
                self.grad_input[:, in_idx, :, :] = convolve(self.grad_output[:, out_idx, :, :], inv_kernel)
        return self.grad_input
        # inverted_paras = np.flipud(np.fliplr(self.parameter_matrix))
        # return convolve(self.grad_output, inverted_paras, mode='constant', cval=0.)

    def create_output_shifts(self, output_channel):
        N = self.kernel_spread
        w11 = np.pad(output_channel[:, :-N, :-N], ((0, 0), (N, 0), (N, 0)))  # shift down right
        w12 = np.pad(output_channel[:, :-N, :], ((0, 0), (N, 0), (0, 0)))  # shift down
        w13 = np.pad(output_channel[:, :-N, N:], ((0, 0), (N, 0), (0, N)))  # shift down left
        w21 = np.pad(output_channel[:, :, :-N], ((0, 0), (0, 0), (N, 0)))  # shift right
        w22 = np.pad(output_channel[:, :, :], ((0, 0), (0, 0), (0, 0)))  # no shift
        w23 = np.pad(output_channel[:, :, N:], ((0, 0), (0, 0), (0, N)))  # shift left
        w31 = np.pad(output_channel[:, N:, :-N], ((0, 0), (0, N), (N, 0)))  # shift right up
        w32 = np.pad(output_channel[:, N:, :], ((0, 0), (0, N), (0, 0)))  # shift up
        w33 = np.pad(output_channel[:, N:, N:], ((0, 0), (0, N), (0, N)))  # shift left up
        shifts = np.stack([w11, w12, w13, w21, w22, w23, w31, w32, w33])
        return shifts

    def eval_parameter_gradient(self):
        num_input_channels = self.input.shape[1]
        num_output_channels = self.grad_output.shape[1]
        para_grads = np.zeros((num_input_channels, num_output_channels, 3, 3))
        for output_channel_idx in range(num_output_channels):
            shifted_output = self.create_output_shifts(self.grad_output[:, output_channel_idx, :, :])
            for input_channel_idx in range(num_input_channels):
                products = shifted_output * self.input[:, input_channel_idx, :, :]
                sum_axes = tuple(range(1, len(products.shape)))
                sub_grads = products.sum(axis=sum_axes).reshape((3, 3))
                para_grads[input_channel_idx, output_channel_idx, :, :] = sub_grads
        return para_grads


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


class ConvLayer(Layer):
    def __init__(self, n_channels, activation_func, flatten=True):
        super().__init__(n_neurons=0)
        self.n_channels = n_channels
        self.input = None
        self.output = None
        self.activation_func = activation_func

        self.flatten = flatten
        self._input_known = False

    def set_up(self):
        num_input_channels, num_output_channels = self.input.shape[1], self.n_channels
        weights = np.random.random((num_input_channels, num_output_channels, 3, 3))
        self.ops.append(ConvChannel(weights))
        self.ops.append(self.activation_func)
        if self.flatten:
            self.ops.append(Flatten())


class PoolingLayer(Layer):
    def __init__(self, pool_size=2, flatten=False):
        super().__init__(n_neurons=0)
        self.pool_size = pool_size
        self.flatten = flatten

    def set_up(self):
        self.ops.append(Pool(pool_size=self.pool_size))
        if self.flatten:
            self.ops.append(Flatten())

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
        self.input = None
        self.output = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_):
        self.input = input_
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
        self.batch_callbacks = []
        self.epoch_callbacks = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def add_batch_callback(self, callback):
        assert isinstance(callback, Callback)
        self.batch_callbacks.append(callback)

    def add_epoch_callback(self, callback):
        assert isinstance(callback, Callback)
        self.epoch_callbacks.append(callback)

    def optimize(self, x_train, y_train, x_test, y_test, epochs=1, batch_size=100):
        from utils import create_batches
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        for epoch in range(epochs):
            for X, y in create_batches(x_train, y_train, batch_size):
                loss = self.nn.eval(X, y)
                self.optimizer.optimization_step()
                print(loss)
                for callback in self.batch_callbacks:
                    callback.step()
            for callback in self.epoch_callbacks:
                callback.step()


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


class Callback:
    def step(self):
        raise NotImplementedError


class ChannelVisualiser(Callback):
    def __init__(self, convLayer: ConvLayer):
        self.layer = convLayer

    def step(self):
        import matplotlib.pyplot as plt
        batch_no = 4
        for parameter in self.layer.parameters:
            print(parameter)
        for channel_index in range(self.layer.input.shape[1]):
            plt.figure()
            plt.imshow(self.layer.input[batch_no, channel_index, :, :])
        for channel_index in range(self.layer.n_channels):
            feature_map = self.layer.ops[0].output[batch_no, channel_index, :, :]
            plt.figure()
            plt.imshow(feature_map)
        plt.show()


class LossHistory(Callback):
    def __init__(self, neural_net: NN, avg_over=False):
        super().__init__()
        self.neural_net = neural_net
        self.loss_hist = []
        self.avg_over = avg_over
        self.averaged_loss = []

    def step(self):
        self.loss_hist.append(self.neural_net.loss_func.output)
        if self.avg_over:
            self.averaged_loss.append(np.average(self.loss_hist[-self.avg_over:]))


class TrainAccuracy(Callback):
    def __init__(self, neural_net: NN, x_test, y_test):
        self.nn = neural_net
        self.x_test = x_test
        self.y_test = y_test

    def step(self):
        prediction = np.zeros_like(self.nn.loss_func.simulated)
        for out, highest_prob_index in zip(prediction, np.argmax(self.nn.loss_func.simulated, axis=1)):
            out[highest_prob_index] = 1
        accuracy = np.sum(prediction * self.nn.loss_func.true) / self.nn.input.shape[0] * 100.
        print(f'train accuracy={accuracy}%')
        self.nn.eval(self.x_test, self.y_test)
        prediction = np.zeros_like(self.nn.loss_func.simulated)
        for out, highest_prob_index in zip(prediction, np.argmax(self.nn.loss_func.simulated, axis=1)):
            out[highest_prob_index] = 1
        accuracy = np.sum(prediction * self.nn.loss_func.true) / self.nn.input.shape[0] * 100.
        print(f'test accuracy={accuracy}%')


class OneTimeTrigger(Callback):
    def __init__(self, value_func, condition):
        self.value_func = value_func
        self.condition = condition
        self.has_triggered = False

    def check_trigger(self):
        if not self.has_triggered:
            if self.condition(self.value_func()):
                self.has_triggered = True
                return True
        return False

    def step(self):
        raise NotImplementedError


class Trigger(Callback):
    def __init__(self, value_func, condition):
        self.value_func = value_func
        self.condition = condition

    def check_trigger(self):
        return self.condition(self.value_func())

    def step(self):
        raise NotImplementedError


class CompareDigitsTrigger(Trigger):
    def __init__(self, value_func, condition, neural_net: NN):
        super().__init__(value_func, condition)
        self.nn = neural_net

    def step(self):
        import matplotlib.pyplot as plt
        if self.check_trigger():
            for batch_no, probs in enumerate(self.nn.loss_func.simulated):
                over = [i for i, prob in enumerate(probs) if prob > 0.75]
                if len(over) > 1:
                    print(over)
                    plt.figure()
                    plt.imshow(np.reshape(self.nn.input[batch_no], (28, 28)))
                    plt.show()


