import pytest
import numpy as np
import NN

import activation_functions as af
import parameterless_operations as plo
import parameterized_operations as po
import layers
import loss_functions as loss_fs


class TestOperations:
    def test_ApplyWeights_forward(self):
        valid_input = np.random.random((5, 4))
        invalid_input = np.random.random((2, 5))
        weights = np.random.random((4, 3))
        op = po.ApplyWeights(weights)
        rv = op.forward(valid_input)
        assert rv.shape == (5, 3)
        with pytest.raises(ValueError):
            op.forward(invalid_input)

    def test_ApplyWeights_backward(self):
        valid_input = np.random.random((5, 4))
        valid_output_grad = np.random.random((5, 3))
        invalid_output_grad = np.random.random((3, 5))
        weights = np.random.random((4, 3))
        op = po.ApplyWeights(weights)
        op.forward(valid_input)
        rv = op.backward(valid_output_grad)
        assert rv.shape == valid_input.shape
        with pytest.raises(ValueError):
            op.backward(invalid_output_grad)

    def test_ApplyBiases_forward(self):
        input_ = np.random.random((5, 3))
        biases = np.random.random((1, 3))
        op = po.ApplyBiases(biases)
        rv = op.forward(input_)
        assert rv.shape == input_.shape
        all(np.testing.assert_array_equal(r[np.newaxis, :], i+biases) for r, i in zip(rv, input_))

    def test_ApplyBiases_backward(self):
        input_ = np.random.random((5, 3))
        biases = np.random.random((1, 3))
        output_grad = np.random.random((5, 3))
        op = po.ApplyBiases(biases)
        op.forward(input_)
        input_grad = op.backward(output_grad)
        assert output_grad.shape == input_.shape


    def test_Sigmoid_forward(self):
        input_ = np.random.random((5, 4))
        op = af.Sigmoid()
        rv = op.forward(input_)
        assert rv.shape == input_.shape

    def test_Sigmoid_backward(self):
        input_ = np.random.random((5, 4))
        output_grad = np.random.random((5, 4))
        op = af.Sigmoid()
        op.forward(input_)
        rv = op.backward(output_grad)
        assert rv.shape == input_.shape

    def test_Conv_reshape_input(self):
        input_ = np.random.random((3, 4, 7, 8))
        op = po.Conv2D(np.array([]))
        rv = op._create_patches(input_, 5)
        assert rv.shape == (56, 3, 4, 5, 5)  # (img_pixel, batch_size, input channels, kernel_size, kernel_size)

    def test_Conv2D_forward(self):
        input_ = np.random.random((3, 4, 7, 8))
        weight_kernel = np.random.random((4, 5, 5, 5))
        op = po.Conv2D(weight_kernel)
        output = op.forward(input_)
        assert output.shape == (3, 5, 7, 8)

    def test_Conv2D_backward(self):
        input_ = np.random.random((3, 4, 7, 8))
        output_grad = np.random.random((3, 6, 7, 8))
        weight_kernel = np.random.random((4, 6, 5, 5))
        op = po.Conv2D(weight_kernel)
        op.forward(input_)
        op.grad_output = output_grad
        op.eval_parameter_gradient()
        op.eval_input_gradient()

    def test_Pool(self):
        batch_size, num_input_channels, num_output_channels, x_px, y_px = 3, 4, 5, 9, 12
        input_ = np.random.random((batch_size, num_input_channels, x_px, y_px))
        op = plo.Pool(2)
        op.forward(input_)

    def test_Dropout_forward(self):
        np.random.seed(1)
        keep_prob = 0.75
        batch_size, num_input_channels, num_output_channels, x_px, y_px = 3, 4, 5, 9, 12
        input_ = np.random.random((batch_size, num_input_channels, x_px, y_px))
        output_grad = np.random.random((batch_size, num_input_channels, x_px, y_px))
        op = plo.Dropout(keep_probability=keep_prob)
        regular_output = op.forward(input_, training_run=False)
        assert regular_output.shape == input_.shape
        assert np.sum(regular_output == 0) == 0
        assert np.isclose(np.sum(regular_output), np.sum(input_) * keep_prob)

        regular_input_grad = op.backward(output_grad)
        assert regular_input_grad.shape == input_.shape
        assert np.sum(regular_input_grad == 0) == 0
        assert np.isclose(np.sum(regular_input_grad), np.sum(output_grad))

        train_output = op.forward(input_, training_run=True)
        assert train_output.shape == input_.shape
        assert np.sum(train_output == 0) != 0


class TestLayer:
    def test_DenseLayer_forward(self):
        input_ = np.random.random((5, 4))
        layer = layers.DenseLayer(3, activation_func=af.Sigmoid())
        rv = layer.forward(input_)
        assert rv.shape == (5, 3)

    def test_DenseLayer_backward(self):
        input_ = np.random.random((5, 4))
        valid_output_grad = np.random.random((5, 3))
        layer = layers.DenseLayer(3, activation_func=af.Sigmoid())
        layer.forward(input_)
        rv = layer.backward(valid_output_grad)
        assert rv.shape == input_.shape


class TestMeanSquaredLoss:
    def test_forward(self):
        loss = loss_fs.MeanSquaredLoss()
        simulated = np.full((5, 3), 10)
        true = np.full((5, 3), 8)
        rv = loss.forward(simulated, true)
        assert rv == 3 * 4

    def test_backward(self):
        loss = loss_fs.MeanSquaredLoss()
        simulated = np.full((1, 5, 3), 10)
        true = np.full((1, 5, 3), 8)
        with pytest.raises(TypeError):
            loss.backward()
        loss.forward(simulated, true)
        rv = loss.backward()
        np.testing.assert_array_equal(rv, np.full((1, 5, 3), 4.))


class TestSoftmaxCrossEntropyLoss:
    def test_softmax(self):
        simulated = np.random.random((5, 3))
        true = np.random.random((5, 3))
        loss = loss_fs.SoftMaxCrossEntropyLoss()
        loss.forward(simulated, true)
        rv = loss.softmax()
        assert all(np.isclose(line.sum(), 1) for line in rv)

    def test_forward(self):
        simulated = np.random.random((5, 3))
        loss = loss_fs.SoftMaxCrossEntropyLoss()


    def test_backward(self):
        pass


class TestNN:
    def test_forward(self):
        input_ = np.random.random((5, 3))
        nn = NN.NN(loss_func=loss_fs.MeanSquaredLoss())
        nn.add_layer(layers.DenseLayer(n_neurons=5, activation_func=af.Sigmoid()))
        nn.add_layer(layers.DenseLayer(n_neurons=7, activation_func=af.Sigmoid()))
        rv = nn.forward(input_)
        assert rv.shape == (5, 7)

    def test_backward(self):
        input_ = np.random.random((5, 3))
        output_grad = np.random.random((5, 7))
        nn = NN.NN(loss_func=loss_fs.MeanSquaredLoss())
        nn.add_layer(layers.DenseLayer(n_neurons=4, activation_func=af.Sigmoid()))
        nn.add_layer(layers.DenseLayer(n_neurons=7, activation_func=af.Sigmoid()))
        nn.forward(input_)
        nn.backward(output_grad)

    def test_eval(self):
        input_ = np.random.random((5, 3))
        true = np.random.random((5, 4))
        nn = NN.NN(loss_func=loss_fs.MeanSquaredLoss())
        nn.add_layer(layers.DenseLayer(n_neurons=8, activation_func=af.Sigmoid()))
        nn.add_layer(layers.DenseLayer(n_neurons=4, activation_func=af.Sigmoid()))
        loss = nn.eval(input_, true)
        assert loss > 0
        assert list(nn.parameters) != []
        assert list(nn.parameter_gradients) != []