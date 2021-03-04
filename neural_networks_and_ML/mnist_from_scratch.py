import NN

from mnist_data import x_train as X
from mnist_data import y_train_reformatted as y

# ===================================================================================


nn = NN.NN(loss_func=NN.MeanSquaredLoss())
# nn = NN.NN(loss_func=NN.SoftMaxCrossEntropyLoss(softmax_clip_eps=1e-7))
nn.add_layer(NN.DenseLayer(128, NN.Sigmoid(), weight_initialisation='glorot'))
nn.add_layer(NN.DenseLayer(10, NN.Sigmoid(), weight_initialisation='glorot'))

# optimizer = NN.SGD(learning_rate=0.01)
optimizer = NN.MomentumSGD(learning_rate=0.01, momentum=0.8)
trainer = NN.Trainer(nn, optimizer)

trainer.optimize(X, y, batch_size=50, epochs=50)
