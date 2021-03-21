import numpy as np
from scipy.special import logsumexp


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


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class SoftMaxCrossEntropyLoss(Loss):
    def __init__(self, softmax_clip_eps=1e-7):
        super().__init__()
        self.softmax_clip_eps = softmax_clip_eps
        self.sm = None

    # def softmax(self):
    #     a = np.exp(self.simulated - np.max(self.simulated, axis=1, keepdims=True))
    #     rv = a / np.sum(a, axis=1, keepdims=True)
    #     return rv

    def softmax(self, x, axis=1):
        a = np.exp(x - np.max(x, axis=1, keepdims=True))
        # v = a / np.sum(a, axis=1, keepdims=True)
        return a / np.sum(a, axis=1, keepdims=True)

    def eval_output(self):
        self.sm = self.softmax(self.simulated)
        if self.softmax_clip_eps is not None:
            self.sm = np.clip(self.sm, self.softmax_clip_eps, 1. - self.softmax_clip_eps)
        # val = -self.true * np.log(sm) - (1 - self.true) * np.log(1 - sm)
        loss = -self.true * np.log(self.sm) - (1. - self.true) * np.log(1. - self.sm)
        return np.sum(loss)

    def eval_input_gradient(self):
        return self.sm - self.true
