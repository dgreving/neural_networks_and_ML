import numpy as np
from operations_base import ParameterOperation


class ConvChannel(ParameterOperation):
    """Kind of the pendant to a single neuron containing a single operation in the case of a DenseLayer...(?)"""

    def __init__(self, weight_kernel):
        """
        weight_kernel has to be of shape (M, N, x_px, y_px), with M being the number of input channels and
        N being the number of output channels, i.e. every channel of the input gets convoluted with a separate
        kernel to produce exactly one output channel. x_px and y_px are the lateral dimensions of each channel.
        """
        super().__init__(weight_kernel)

    def eval_output(self):
        batch_size = self.input.shape[0]
        n_input_channels = self.input.shape[1]
        n_output_channels = self.parameter_matrix.shape[1]  # para.shape = (input_channels, output_channels, x, y)

        self.output = np.zeros((batch_size, n_output_channels, self.input.shape[-2], self.input.shape[-1]))
        for out_idx in range(n_output_channels):
            for in_idx in range(n_input_channels):
                kernel = self.parameter_matrix[np.newaxis, in_idx, out_idx, :, :]
                self.output[:, out_idx, :, :] += convolve(self.input[:, in_idx, :, :], kernel)
        return self.output

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

    def _shift(self, output, x_shift, y_shift=0):
        xstart = None if x_shift <= 0 else x_shift
        xend = None if x_shift >= 0 else x_shift  # np.minimum(x_shift, 0)
        ystart = None if y_shift <= 0 else y_shift
        yend = None if y_shift >= 0 else y_shift  # np.minimum(y_shift, 0)
        xpad0 = xstart or 0
        xpad1 = xend or 0
        ypad0 = ystart or 0
        ypad1 = yend or 0
        return np.pad(output[..., xstart:xend, ystart:yend], ((0, 0), (0, 0), (-xpad1, xpad0), (-ypad1, ypad0)))

    def eval_parameter_gradient(self):
        para_grad = np.zeros_like(self.parameter_matrix)
        para_dim_x, para_dim_y = self.parameter_matrix.shape[-2], self.parameter_matrix.shape[-1]
        for i, x_shift in enumerate(range(-para_dim_x//2, para_dim_x//2 + 1)):
            for j, y_shift in enumerate(range(-para_dim_y//2, para_dim_y//2 + 1)):
                print(i, j)
                print(self.input.shape, self.grad_output.shape)
                one_pos = np.sum(
                    self._shift(self.input, x_shift, y_shift) * self.grad_output, axis=(0, -2, -1)
                )
                para_grad[..., i, j] = one_pos
        self.grad_parameters = para_grad
        return self.grad_parameters