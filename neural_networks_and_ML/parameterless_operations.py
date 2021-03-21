import numpy as np
from operations_base import Operation


class Dropout(Operation):
    def __init__(self, keep_probability=0.75):
        super().__init__()
        self.keep_probability = keep_probability
        self.mask = None

    def eval_output(self):
        if not self.training_run:
            print('never. EVER!')
            self.mask = np.array([1.])
            return self.input * self.keep_probability
        else:
            self.mask = np.random.binomial(1, self.keep_probability, size=self.input.shape)

            return self.input * self.mask

    def eval_input_gradient(self):
        return self.mask * self.grad_output


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
