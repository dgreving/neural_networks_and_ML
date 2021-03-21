import numpy as np
from NN import NN
from layers import ConvLayer


def create_batches(x, y, size=100):
    start, end = 0, size
    while end < x.shape[0]:
        yield x[start:end, :], y[start:end]
        start += size
        end += size


class Callback:
    def step(self):
        raise NotImplementedError


class ProgressBar(Callback):
    def __init__(self, total, incr=1, num_chars=20):
        self.total = total
        self.incr = incr
        self.current = 0
        self.num_chars = num_chars

    def step(self):
        self.current += self.incr
        frac = self.current / self.total
        s = '|'
        s += int(frac * self.num_chars) * '#'
        s += int((1 - frac) * self.num_chars) * ' '
        s += '|'
        print(s, end='\r')
        if self.current >= self.total:
            self.current = 0
            print('')


class SaveModel(Callback):
    def __init__(self, nn):
        self.nn = nn

    def step(self):
        import pickle
        with open('nn.pkl', 'wb') as f:
            pickle.dump(self.nn, f)


class ChannelVisualiser(Callback):
    def __init__(self, convLayer: ConvLayer):
        self.layer = convLayer

    def step(self):
        import matplotlib.pyplot as plt
        batch_no = 6
        for channel_index in range(self.layer.input.shape[1]):
            plt.figure()
            plt.imshow(self.layer.input[batch_no, channel_index, :, :])
        for channel_index in range(self.layer.n_channels):
            feature_map = self.layer.ops[0].output[batch_no, channel_index, :, :]
            kernel = self.layer.ops[0].parameter_matrix[0, channel_index, ...]
            plt.figure()
            plt.imshow(feature_map)
            plt.figure()
            plt.imshow(kernel)
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