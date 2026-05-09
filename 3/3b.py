# Experiment with Different Optimizers

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Function to create model
def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

    ])

    return model


# Adam Optimizer
model_adam = create_model()

model_adam.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with Adam Optimizer")

model_adam.fit(X_train, y_train, epochs=5)

adam_loss, adam_accuracy = model_adam.evaluate(X_test, y_test)

print("Adam Accuracy:", adam_accuracy)


# RMSProp Optimizer
model_rms = create_model()

model_rms.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with RMSProp Optimizer")

model_rms.fit(X_train, y_train, epochs=5)

rms_loss, rms_accuracy = model_rms.evaluate(X_test, y_test)

print("RMSProp Accuracy:", rms_accuracy)