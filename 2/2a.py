# Imports required packages

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# Loads MNIST dataset

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()


# Checks the shape of the datasets

print("Full training set shape:", X_train_full.shape)
print("Test set shape:", X_test.shape)


# Normalizes the data between 0 and 1

X_train_full = X_train_full / 255.
X_test = X_test / 255.


# Splits train dataset into training and validation set

X_train, X_val = X_train_full[:-5000], X_train_full[-5000:]

y_train, y_val = y_train_full[:-5000], y_train_full[-5000:]


# Adds channel dimension

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# Checks updated shape

print(X_train.shape)


# Creates CNN model

tf.random.set_seed(42)

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )

])


# Compiles model

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)


# Fits the model

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)


# Saves the model

model.save("./models/my_mnist_cnn_model.keras")


# Evaluates model

model.evaluate(X_test, y_test)