# Build a Simple Sequential CNN Model for MNIST Classification

# Import required packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Check shape of datasets
print("Training Set Shape:", X_train_full.shape)
print("Test Set Shape:", X_test.shape)

# Normalize the data
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split validation dataset
X_train = X_train_full[:-5000]
X_val = X_train_full[-5000:]

y_train = y_train_full[:-5000]
y_val = y_train_full[-5000:]

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Check updated shape
print("Updated Training Shape:", X_train.shape)

# Create CNN Model
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

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Save model
model.save("my_mnist_cnn_model.keras")

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)