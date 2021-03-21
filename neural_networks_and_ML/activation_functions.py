import numpy as np
from operations_base import Operation


class Linear(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return self.input

    def eval_input_gradient(self):
        return self.grad_output


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return 1. / (1 + np.exp(-self.input))

    def eval_input_gradient(self):
        return self.grad_output * self.output * (1 - self.output)


class Tanh(Operation):
    def __init__(self):
        super().__init__()

    def eval_output(self):
        return np.tanh(self.input)

    def eval_input_gradient(self):
        return self.grad_output * (1 - self.output**2)
