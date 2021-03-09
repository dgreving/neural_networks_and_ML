import NN

from mnist_data import x_train_2D as X
from mnist_data import y_train_reformatted as y
from mnist_data import x_test_2D as x_test
from mnist_data import y_test_reformatted as y_test

# ===================================================================================


nn = NN.NN(loss_func=NN.MeanSquaredLoss())

#nn.add_layer(NN.PoolingLayer(pool_size=2, flatten=False))

convLayer = NN.ConvLayer(n_channels=5, activation_func=NN.Sigmoid(), flatten=False)
nn.add_layer(convLayer)
nn.add_layer(NN.PoolingLayer(pool_size=5, flatten=True))
nn.add_layer(NN.DenseLayer(10, NN.Sigmoid(), weight_initialisation='glorot'))

# optimizer = NN.SGD(learning_rate=0.01)
# optimizer = NN.MomentumSGD(learning_rate=0.001, momentum=0.8)
optimizer = NN.SGD(learning_rate=0.01)
trainer = NN.Trainer(nn, optimizer)

# statistic
loss_history = NN.LossHistory(nn, avg_over=50)
trainer.add_batch_callback(loss_history)

visualiser = NN.ChannelVisualiser(convLayer)
trainer.add_epoch_callback(visualiser)

accuracy = NN.TrainAccuracy(nn, x_test, y_test)
trainer.add_epoch_callback(accuracy)

trigger = NN.CompareDigitsTrigger(lambda: loss_history.averaged_loss[-1], lambda x: x < 0.4, nn)
trainer.add_batch_callback(trigger)

trainer.optimize(X, y, x_test, y_test, batch_size=50, epochs=3)

