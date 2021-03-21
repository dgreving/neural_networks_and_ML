# import numpy as np
import cupy as np
from operations_base import ParameterOperation


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


class Conv2D(ParameterOperation):
    def __init__(self, kernel):
        super().__init__(parameter_matrix=kernel)
        self.kernel_size = kernel.shape[-1]

    @property
    def batch_size(self):
        return self.input.shape[0]

    @property
    def img_dims(self):
        return self.input.shape[-2:]

    @property
    def num_input_channels(self):
        return self.parameter_matrix.shape[0]

    @property
    def num_output_channels(self):
        return self.parameter_matrix.shape[1]

    def _create_patches(self, ndarr, kernel_size):
        dim1, dim2 = ndarr.shape[-2:]
        pw = kernel_size // 2  # pad_width
        padded_input = np.pad(ndarr, ((0, 0), (0, 0), (pw, pw), (pw, pw)))
        patches = []
        for i in range(dim1):
            for j in range(dim2):
                patches.append(padded_input[..., i: i + kernel_size, j: j+kernel_size])
        # shape patches until here: (img_pixels, batch_size, n_input_channels, kernel_size, kernel_size)
        return np.stack(patches)

    def _reshape_parameters(self):
        # parameter shape is (num_input_channel, num_output_channels, kernel_size, kernel_size)
        # reshape so that first dimension is proportional to number of input channels
        num_output_channels = self.parameter_matrix.shape[1]
        reshaped_parameter = self.parameter_matrix.transpose((0, 2, 3, 1)).reshape((-1, num_output_channels))
        # shape reshaped parameters: (kernel_size**2 * number_input_channel, number_output_channels)
        return reshaped_parameter

    def eval_output(self):
        dim1, dim2 = self.img_dims
        patches = self._create_patches(self.input, self.kernel_size)
        reshaped_input = patches.transpose((1, 0, 2, 3, 4)).reshape((self.batch_size, len(patches), -1))
        # shape reshaped_input: (batch_size, number_of_patches, kernel_size**2 * number_input_channels)
        reshaped_parameters = self._reshape_parameters()
        reshaped_output = np.matmul(reshaped_input, reshaped_parameters)
        self.output = (
            reshaped_output
            .reshape((self.batch_size, dim1, dim2, self.num_output_channels))
            .transpose((0, 3, 1, 2))
        )
        # shape output: (batch_size, num_output_channels, img_dim1, img_dim2)
        return self.output

    def eval_parameter_gradient(self):
        num_output_channels = self.grad_output.shape[1]
        img_dim1, img_dim2 = self.img_dims
        patches = self._create_patches(self.input, self.kernel_size)
        # reshape patches into 2D matrix, containing all input pixels (batches and 2D img) on each row
        reshaped_patches = (
            patches
            .transpose((2, 3, 4, 1, 0))
            .reshape(self.num_input_channels * self.kernel_size**2, self.batch_size * img_dim1 * img_dim2)
        )
        reshaped_output_grad = (
            self.grad_output
            .transpose((0, 2, 3, 1))
            .reshape((self.batch_size * img_dim1 * img_dim2, num_output_channels))
        )
        parameter_grad_reshaped = np.matmul(reshaped_patches, reshaped_output_grad)
        parameter_grad = (
            parameter_grad_reshaped
            .reshape((self.num_input_channels, self.kernel_size, self.kernel_size, num_output_channels))
            .transpose((0, 3, 1, 2))
        )
        self.grad_parameters = parameter_grad
        return parameter_grad

    def eval_input_gradient(self):
        output_grad_patches = self._create_patches(self.grad_output, self.kernel_size)
        img_dim1, img_dim2 = self.img_dims
        # (56, 3, 6, 5, 5)
        output_grad_patches = (
            output_grad_patches
            .transpose((0, 1, 2, 3, 4))
            .reshape((self.batch_size * img_dim1 * img_dim2, self.num_output_channels * self.kernel_size**2))
        )
        reshaped_parameters = (
            self.parameter_matrix
            .transpose(1, 2, 3, 0)
            .reshape((self.num_output_channels * self.kernel_size**2, self.num_input_channels))
        )

        reshaped_input_grad = np.matmul(output_grad_patches, reshaped_parameters)

        input_grad = (
            reshaped_input_grad
            .reshape((self.batch_size, img_dim1, img_dim2, self.num_input_channels))
            .transpose((0, 3, 1, 2))
        )
        self.grad_input = input_grad
        return input_grad
