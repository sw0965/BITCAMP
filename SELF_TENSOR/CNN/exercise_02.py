import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
np.set_printoptions(linewidth=200)

import matplotlib.pyplot as plt
plt.imshow(training_images[10])
plt.show()

print(training_labels[10])
print(training_images[10])

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

history = model.compile(optimizer=tf.optimizers.Adam(),
                        loss= 'sparse_categorical_crossentropy',
                        metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)