import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# Create CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )

])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=3
)

# Evaluate baseline model
loss, accuracy = model.evaluate(X_test, y_test)

print("Baseline Accuracy:", accuracy)

# Apply pruning
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

# Compile pruned model
pruned_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train pruned model
pruned_model.fit(
    X_train,
    y_train,
    epochs=2
)

# Evaluate pruned model
loss, pruned_accuracy = pruned_model.evaluate(X_test, y_test)

print("Pruned Model Accuracy:", pruned_accuracy)

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_model = converter.convert()

# Print model size
print(
    "Quantized Model Size:",
    len(quantized_model) / 1024,
    "KB"
)