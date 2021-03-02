from tensorflow.keras.utils import normalize
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = normalize(x_train), normalize(x_test)

x_train = np.reshape(x_train, (60000, 28 * 28))
y_train_reformatted = np.zeros((60000, 10))
for i, correct in enumerate(y_train):
    y_train_reformatted[i, correct] = 1


