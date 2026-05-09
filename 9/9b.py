import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize data
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Generator model
generator = tf.keras.Sequential([

    tf.keras.layers.Dense(
        128,
        activation='relu',
        input_dim=100
    ),

    tf.keras.layers.Dense(
        28 * 28,
        activation='tanh'
    ),

    tf.keras.layers.Reshape((28, 28, 1))

])

# Discriminator model
discriminator = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28,1)),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )

])

# Compile discriminator
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Freeze discriminator
discriminator.trainable = False

# Build GAN
gan = tf.keras.Sequential([
    generator,
    discriminator
])

# Compile GAN
gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Training
epochs = 1000
batch_size = 64

for epoch in range(epochs):

    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)

    real_images = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))

    fake_images = generator.predict(noise)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    discriminator.train_on_batch(real_images, real_labels)

    discriminator.train_on_batch(fake_images, fake_labels)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))

    gan.train_on_batch(noise, real_labels)

    # Print progress
    if epoch % 100 == 0:
        print("Epoch:", epoch)

# Generate image
noise = np.random.normal(0, 1, (1, 100))

generated_image = generator.predict(noise)

plt.imshow(
    generated_image[0, :, :, 0],
    cmap='gray'
)

plt.show()