import NN
import utils

from mnist_data import x_train as X
from mnist_data import y_train_reformatted as y
from mnist_data import x_test
from mnist_data import y_test_reformatted as y_test

# ===================================================================================


nn = NN.NN(loss_func=NN.MeanSquaredLoss())
# nn = NN.NN(loss_func=NN.SoftMaxCrossEntropyLoss(softmax_clip_eps=1e-7))
nn.add_layer(NN.DenseLayer(128, NN.Sigmoid(), weight_initialisation='glorot', dropout=0.8))
nn.add_layer(NN.DenseLayer(10, NN.Sigmoid(), weight_initialisation='glorot'))

# optimizer = NN.SGD(learning_rate=0.01)
optimizer = NN.MomentumSGD(learning_rate=0.01, momentum=0.8)
trainer = NN.Trainer(nn, optimizer)

# statistic
loss_history = utils.LossHistory(nn, avg_over=50)
trainer.add_batch_callback(loss_history)

accuracy = utils.TrainAccuracy(nn, x_test, y_test)
trainer.add_epoch_callback(accuracy)

# trigger = utils.CompareDigitsTrigger(lambda: loss_history.averaged_loss[-1], lambda x: x < 0.4, nn)
# trainer.add_batch_callback(trigger)

trainer.optimize(X, y, x_test, y_test, batch_size=50, epochs=20)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_history.loss_hist)
plt.plot(loss_history.averaged_loss)
plt.show()
