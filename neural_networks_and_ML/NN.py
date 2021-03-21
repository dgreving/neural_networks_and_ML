from loss_functions import MeanSquaredLoss
from optimizers import Optimizer


# from numpy import ndarray
# class Conv2D(ParameterOperation):
#     def __init__(self, W: ndarray):
#         super().__init__(W)
#         self.param_size = W.shape[2]
#         self.param_pad = self.param_size // 2
#
#     def _pad_1d(self, inp: ndarray) -> ndarray:
#         z = np.array([0])
#         z = np.repeat(z, self.param_pad)
#         return np.concatenate([z, inp, z])
#
#     def _pad_1d_batch(self,
#                       inp: ndarray) -> ndarray:
#         outs = [self._pad_1d(obs) for obs in inp]
#         return np.stack(outs)
#
#     def _pad_2d_obs(self,
#                     inp: ndarray):
#         '''
#         Input is a 2 dimensional, square, 2D Tensor
#         '''
#         inp_pad = self._pad_1d_batch(inp)
#
#         other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))
#
#         return np.concatenate([other, inp_pad, other])
#
#
#     # def _pad_2d(self,
#     #             inp: ndarray):
#     #     '''
#     #     Input is a 3 dimensional tensor, first dimension batch size
#     #     '''
#     #     outs = [self._pad_2d_obs(obs, self.param_pad) for obs in inp]
#     #
#     #     return np.stack(outs)
#
#     def _pad_2d_channel(self,
#                         inp: ndarray):
#         '''
#         inp has dimension [num_channels, image_width, image_height]
#         '''
#         return np.stack([self._pad_2d_obs(channel) for channel in inp])
#
#     def _get_image_patches(self,
#                            input_: ndarray):
#         imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
#         patches = []
#         img_height = imgs_batch_pad.shape[2]
#         for h in range(img_height-self.param_size+1):
#             for w in range(img_height-self.param_size+1):
#                 patch = imgs_batch_pad[:, :, h:h+self.param_size, w:w+self.param_size]
#                 patches.append(patch)
#         return np.stack(patches)
#
#     def eval_output(self,
#                 inference: bool = False):
#         '''
#         conv_in: [batch_size, channels, img_width, img_height]
#         param: [in_channels, out_channels, fil_width, fil_height]
#         '''
#     #     assert_dim(obs, 4)
#     #     assert_dim(param, 4)
#         batch_size = self.input.shape[0]
#         img_height = self.input.shape[2]
#         img_size = self.input.shape[2] * self.input.shape[3]
#         patch_size = self.parameter_matrix.shape[0] * self.parameter_matrix.shape[2] * self.parameter_matrix.shape[3]
#
#         patches = self._get_image_patches(self.input)
#
#         patches_reshaped = (patches
#                             .transpose(1, 0, 2, 3, 4)
#                             .reshape(batch_size, img_size, -1))
#
#         param_reshaped = (self.parameter_matrix
#                           .transpose(0, 2, 3, 1)
#                           .reshape(patch_size, -1))
#
#         output_reshaped = (
#             np.matmul(patches_reshaped, param_reshaped)
#             .reshape(batch_size, img_height, img_height, -1)
#             .transpose(0, 3, 1, 2))
#
#         return output_reshaped
#
#     def eval_input_gradient(self):
#         output_grad = self.grad_output
#         batch_size = self.input.shape[0]
#         img_size = self.input.shape[2] * self.input.shape[3]
#         img_height = self.input.shape[2]
#
#         output_patches = (self._get_image_patches(output_grad)
#                           .transpose(1, 0, 2, 3, 4)
#                           .reshape(batch_size * img_size, -1))
#
#         param_reshaped = (self.parameter_matrix
#                           .reshape(self.parameter_matrix.shape[0], -1)
#                           .transpose(1, 0))
#
#         return (
#             np.matmul(output_patches, param_reshaped)
#             .reshape(batch_size, img_height, img_height, self.parameter_matrix.shape[0])
#             .transpose(0, 3, 1, 2)
#         )
#
#
#     def eval_parameter_gradient(self):
#         output_grad = self.grad_output
#         batch_size = self.input.shape[0]
#         img_size = self.input.shape[2] * self.input.shape[3]
#         in_channels = self.parameter_matrix.shape[0]
#         out_channels = self.parameter_matrix.shape[1]
#
#         in_patches_reshape = (
#             self._get_image_patches(self.input)
#             .reshape(batch_size * img_size, -1)
#             .transpose(1, 0)
#             )
#
#         out_grad_reshape = (output_grad
#                             .transpose(0, 2, 3, 1)
#                             .reshape(batch_size * img_size, -1))
#
#         return (np.matmul(in_patches_reshape,
#                           out_grad_reshape)
#                 .reshape(in_channels, self.param_size, self.param_size, out_channels)
#                 .transpose(0, 3, 1, 2))


class NN:
    def __init__(self, loss_func=MeanSquaredLoss()):
        self.layers = []
        self.loss_func = loss_func
        self.input = None
        self.output = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_, training_run=False):
        self.input = input_
        for layer in self.layers:
            input_ = layer.forward(input_, training_run=training_run)
        self.output = input_
        return self.output

    def backward(self, loss_gradient):
        output_gradient = loss_gradient
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def eval(self, x, y, training_run=False):
        simulated = self.forward(x, training_run)
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


class Trainer:
    def __init__(self, neural_net: NN, optimizer: Optimizer):
        self.nn = neural_net
        self.optimizer = optimizer
        self.optimizer.add_neural_net(self.nn)
        self.batch_callbacks = []
        self.epoch_callbacks = []
        self.on_finish_callbacks = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def add_batch_callback(self, callback):
        self.batch_callbacks.append(callback)

    def add_epoch_callback(self, callback):
        self.epoch_callbacks.append(callback)

    def add_on_finish_callback(self, callback):
        self.on_finish_callbacks.append(callback)

    def optimize(self, x_train, y_train, x_test, y_test, epochs=1, batch_size=100):
        from utils import create_batches
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        for epoch in range(epochs):
            for X, y in create_batches(x_train, y_train, batch_size):
                loss = self.nn.eval(X, y, training_run=True)
                self.optimizer.optimization_step()
                # print(loss)
                for callback in self.batch_callbacks:
                    callback.step()
            for callback in self.epoch_callbacks:
                callback.step()
        for callback in self.on_finish_callbacks:
            callback.step()
