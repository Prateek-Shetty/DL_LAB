#Implement quantization and pruning techniques in a neural network to reduce its size and computational demands compare results with the baseline models

import tensorflow as tf

import tensorflow_model_optimization as tfmot

import numpy as np

import tempfile


# Load MNIST dataset

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype(np.float32)

    x_test = x_test[..., tf.newaxis].astype(np.float32)

    return (x_train, y_train), (x_test, y_test)


# Define a simple CNN model

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(

            32,

            (3, 3),

            activation='relu',

            input_shape=(28, 28, 1)

        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

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

    return model


# Train baseline model

def train_model(model, x_train, y_train, x_test, y_test):

    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    model.fit(

        x_train,

        y_train,

        epochs=5,

        validation_data=(x_test, y_test)

    )

    return model


# Apply Pruning

def prune_model(model):

    pruning_params = {

        'pruning_schedule':

        tfmot.sparsity.keras.PolynomialDecay(

            initial_sparsity=0.2,

            final_sparsity=0.8,

            begin_step=0,

            end_step=1000

        )

    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(

        model,

        **pruning_params

    )

    pruned_model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    return pruned_model


# Convert to TFLite Model (Quantization)

def quantize_model(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_model = converter.convert()

    return quantized_model


# Evaluate model

def evaluate_model(model, x_test, y_test):

    loss, accuracy = model.evaluate(

        x_test,

        y_test,

        verbose=0

    )

    return accuracy


# Main Execution

(x_train, y_train), (x_test, y_test) = load_data()


# Train baseline model

baseline_model = create_model()

baseline_model = train_model(

    baseline_model,

    x_train,

    y_train,

    x_test,

    y_test

)

baseline_accuracy = evaluate_model(

    baseline_model,

    x_test,

    y_test

)

print(f'Baseline Accuracy: {baseline_accuracy:.4f}')


# Apply pruning and retrain

pruned_model = prune_model(baseline_model)

pruned_model.fit(

    x_train,

    y_train,

    epochs=2,

    validation_data=(x_test, y_test)

)

pruned_accuracy = evaluate_model(

    pruned_model,

    x_test,

    y_test

)

print(f'Pruned Model Accuracy: {pruned_accuracy:.4f}')


# Convert and apply quantization

quantized_model = quantize_model(pruned_model)

print(

    f'Quantized Model Size: {len(quantized_model) / 1024:.2f} KB'

)