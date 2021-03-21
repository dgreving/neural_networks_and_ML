import numpy as np


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
