# Transfer Learning using Fashion MNIST

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


# Create pretrained model

pretrained_model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])

# Compile pretrained model

pretrained_model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

# Train pretrained model

pretrained_model.fit(

    X_train,
    y_train,

    epochs=5

)

# Save pretrained model

pretrained_model.save("pretrained_model.keras")


# Load pretrained model

model = tf.keras.models.load_model("pretrained_model.keras")


# Remove output layer

model.pop()


# Add new output layer

model.add(

    tf.keras.layers.Dense(
        2,
        activation='softmax'
    )

)


# Freeze previous layers

for layer in model.layers[:-1]:

    layer.trainable = False


# Compile transfer learning model

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)


# Train model again

model.fit(

    X_train,
    y_train,

    epochs=3

)


# Evaluate model

model.evaluate(

    X_test,
    y_test

)