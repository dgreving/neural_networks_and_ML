import NN
import utils
import layers
import optimizers
from activation_functions import Sigmoid, Tanh, Linear
from loss_functions import SoftMaxCrossEntropyLoss

from mnist_data import x_train_2D as X
from mnist_data import y_train_reformatted as y
from mnist_data import x_test_2D as x_test
from mnist_data import y_test_reformatted as y_test

# ===================================================================================


nn = NN.NN(loss_func=SoftMaxCrossEntropyLoss())

convLayer = layers.ConvLayer(
    n_channels=9,
    kernel_size=9,
    weight_init='glorot',
    activation_func=Tanh(),
    flatten=True,
    dropout=0.8,
)
# pool = layers.PoolingLayer(pool_size=7, flatten=True)
dense = layers.DenseLayer(10, Linear(), weight_initialisation='glorot')

nn.add_layer(convLayer)
# nn.add_layer(pool)
nn.add_layer(dense)

optimizer = optimizers.MomentumSGD(learning_rate=0.01, momentum=0.90)
trainer = NN.Trainer(nn, optimizer)

# statistic
batch_size = 50

bar = utils.ProgressBar(len(X), batch_size)
trainer.add_batch_callback(bar)

# loss_history = utils.LossHistory(nn, avg_over=50)
# trainer.add_batch_callback(loss_history)

accuracy = utils.TrainAccuracy(nn, x_test[:500:], y_test[:500])
trainer.add_epoch_callback(accuracy)

visualiser = utils.ChannelVisualiser(convLayer)
trainer.add_on_finish_callback(visualiser)

trainer.optimize(X, y, x_test, y_test, batch_size=batch_size, epochs=6)
