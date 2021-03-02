import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = tf.keras.utils.normalize(x_train), tf.keras.utils.normalize(x_test)

model = tf.keras.models.Sequential()

input_layer = tf.keras.layers.Flatten()
layer2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
layer3 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
output_layer = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)

for layer in (input_layer, layer2, layer3, output_layer):
    model.add(layer)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(x_train, y_train, epochs=3)

plt.figure()
plt.imshow(x_train[0])
plt.show()