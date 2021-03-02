import numpy as np
import NN
from utils import create_batches

from mnist_data import x_train as X
from mnist_data import y_train_reformatted as y

# ===================================================================================


nn = NN.NN(loss_func=NN.MeanSquaredLoss())
nn.add_layer(NN.DenseLayer(128, NN.Sigmoid()))
nn.add_layer(NN.DenseLayer(10, NN.Sigmoid()))


while True:
    for X1, y1 in create_batches(X, y):
        loss = nn.eval(X1, y1)
        print(loss)
        for parameter, parameter_grad in zip(nn.parameters, nn.parameter_gradients):
            parameter -= 0.0001 * parameter_grad
