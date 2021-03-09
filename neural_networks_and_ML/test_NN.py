import pytest
import numpy as np
import NN


class TestOperations:
    def test_ApplyWeights_forward(self):
        valid_input = np.random.random((5, 4))
        invalid_input = np.random.random((2, 5))
        weights = np.random.random((4, 3))
        op = NN.ApplyWeights(weights)
        rv = op.forward(valid_input)
        assert rv.shape == (5, 3)
        with pytest.raises(ValueError):
            op.forward(invalid_input)

    def test_ApplyWeights_backward(self):
        valid_input = np.random.random((5, 4))
        valid_output_grad = np.random.random((5, 3))
        invalid_output_grad = np.random.random((3, 5))
        weights = np.random.random((4, 3))
        op = NN.ApplyWeights(weights)
        op.forward(valid_input)
        rv = op.backward(valid_output_grad)
        assert rv.shape == valid_input.shape
        with pytest.raises(ValueError):
            op.backward(invalid_output_grad)

    def test_ApplyBiases_forward(self):
        input_ = np.random.random((5, 3))
        biases = np.random.random((1, 3))
        op = NN.ApplyBiases(biases)
        rv = op.forward(input_)
        assert rv.shape == input_.shape
        all(np.testing.assert_array_equal(r[np.newaxis, :], i+biases) for r, i in zip(rv, input_))

    def test_ApplyBiases_backward(self):
        input_ = np.random.random((5, 3))
        biases = np.random.random((1, 3))
        output_grad = np.random.random((5, 3))
        op = NN.ApplyBiases(biases)
        op.forward(input_)
        input_grad = op.backward(output_grad)
        assert output_grad.shape == input_.shape


    def test_Sigmoid_forward(self):
        input_ = np.random.random((5, 4))
        op = NN.Sigmoid()
        rv = op.forward(input_)
        assert rv.shape == input_.shape

    def test_Sigmoid_backward(self):
        input_ = np.random.random((5, 4))
        output_grad = np.random.random((5, 4))
        op = NN.Sigmoid()
        op.forward(input_)
        rv = op.backward(output_grad)
        assert rv.shape == input_.shape

    def test_ConvChannel_forward(self):
        batch_size, num_channels, x_px, y_px = 3, 4, 9, 12
        num_out = 5
        input_ = np.random.random((batch_size, num_channels, x_px, y_px))
        weight_kernel = np.random.random((num_channels, num_out, 3, 3))
        op = NN.ConvChannel(weight_kernel)
        rv = op.forward(input_)
        assert rv.shape == (batch_size, num_out, x_px, y_px)

    def test_ConvChannel_backward(self):
        batch_size, num_input_channels, num_output_channels, x_px, y_px = 3, 4, 5, 9, 12
        input_ = np.random.random((batch_size, num_input_channels, x_px, y_px))
        output_grad = np.random.random((batch_size, num_output_channels, x_px, y_px))
        weight_kernel = np.random.random((num_input_channels, num_output_channels, 3, 3))
        op = NN.ConvChannel(weight_kernel)
        op.forward(input_)
        rv = op.backward(output_grad)
        assert rv.shape == input_.shape
        assert op.grad_parameters.shape == weight_kernel.shape

    def test_Pool(self):
        batch_size, num_input_channels, num_output_channels, x_px, y_px = 3, 4, 5, 9, 12
        input_ = np.random.random((batch_size, num_input_channels, x_px, y_px))
        op = NN.Pool(2)
        op.forward(input_)


class TestLayer:
    def test_DenseLayer_forward(self):
        input_ = np.random.random((5, 4))
        layer = NN.DenseLayer(3, activation_func=NN.Sigmoid())
        rv = layer.forward(input_)
        assert rv.shape == (5, 3)

    def test_DenseLayer_backward(self):
        input_ = np.random.random((5, 4))
        valid_output_grad = np.random.random((5, 3))
        layer = NN.DenseLayer(3, activation_func=NN.Sigmoid())
        layer.forward(input_)
        rv = layer.backward(valid_output_grad)
        assert rv.shape == input_.shape


class TestMeanSquaredLoss:
    def test_forward(self):
        loss = NN.MeanSquaredLoss()
        simulated = np.full((5, 3), 10)
        true = np.full((5, 3), 8)
        rv = loss.forward(simulated, true)
        assert rv == 3 * 4

    def test_backward(self):
        loss = NN.MeanSquaredLoss()
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
        loss = NN.SoftMaxCrossEntropyLoss()
        loss.forward(simulated, true)
        rv = loss.softmax()
        assert all(np.isclose(line.sum(), 1) for line in rv)

    def test_forward(self):
        simulated = np.random.random((5, 3))
        loss = NN.SoftMaxCrossEntropyLoss()


    def test_backward(self):
        pass


class TestNN:
    def test_forward(self):
        input_ = np.random.random((5, 3))
        nn = NN.NN(loss_func=NN.MeanSquaredLoss())
        nn.add_layer(NN.DenseLayer(n_neurons=5, activation_func=NN.Sigmoid()))
        nn.add_layer(NN.DenseLayer(n_neurons=7, activation_func=NN.Sigmoid()))
        rv = nn.forward(input_)
        assert rv.shape == (5, 7)

    def test_backward(self):
        input_ = np.random.random((5, 3))
        output_grad = np.random.random((5, 7))
        nn = NN.NN(loss_func=NN.MeanSquaredLoss())
        nn.add_layer(NN.DenseLayer(n_neurons=4, activation_func=NN.Sigmoid()))
        nn.add_layer(NN.DenseLayer(n_neurons=7, activation_func=NN.Sigmoid()))
        nn.forward(input_)
        nn.backward(output_grad)

    def test_eval(self):
        input_ = np.random.random((5, 3))
        true = np.random.random((5, 4))
        nn = NN.NN(loss_func=NN.MeanSquaredLoss())
        nn.add_layer(NN.DenseLayer(n_neurons=8, activation_func=NN.Sigmoid()))
        nn.add_layer(NN.DenseLayer(n_neurons=4, activation_func=NN.Sigmoid()))
        loss = nn.eval(input_, true)
        assert loss > 0
        assert list(nn.parameters) != []
        assert list(nn.parameter_gradients) != []