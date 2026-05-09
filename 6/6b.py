import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add noise
noise_factor = 0.3

X_train_noisy = X_train + noise_factor * np.random.randn(*X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.randn(*X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Display noisy image
plt.imshow(X_train_noisy[0], cmap='gray')

# Build Autoencoder model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        16,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D((2,2), padding='same'),

    tf.keras.layers.Conv2D(
        8,
        (3,3),
        activation='relu',
        padding='same'
    ),

    tf.keras.layers.MaxPooling2D((2,2), padding='same'),

    tf.keras.layers.Conv2DTranspose(
        8,
        (3,3),
        strides=2,
        activation='relu',
        padding='same'
    ),

    tf.keras.layers.Conv2DTranspose(
        16,
        (3,3),
        strides=2,
        activation='relu',
        padding='same'
    ),

    tf.keras.layers.Conv2D(
        1,
        (3,3),
        activation='sigmoid',
        padding='same'
    )

])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train model
model.fit(

    X_train_noisy.reshape(-1,28,28,1),

    X_train.reshape(-1,28,28,1),

    epochs=5,

    batch_size=128

)

# Evaluate model
model.evaluate(

    X_test_noisy.reshape(-1,28,28,1),

    X_test.reshape(-1,28,28,1)

)

# Predict denoised images
predicted = model.predict(

    X_test_noisy[:10].reshape(-1,28,28,1)

)

# Display predicted image
plt.imshow(predicted[0].reshape(28,28), cmap='gray')