#Implement a simple GAN to generate images from random noise (e.g., MNIST digit generation).


import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import numpy as np

import matplotlib.pyplot as plt


# Load and preprocess MNIST dataset

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype(np.float32) - 127.5) / 127.5

x_train = np.expand_dims(x_train, axis=-1)


# Define Generator

def build_generator():

    model = Sequential([

        Dense(256, input_dim=100),

        LeakyReLU(0.2),

        BatchNormalization(),

        Dense(512),

        LeakyReLU(0.2),

        BatchNormalization(),

        Dense(1024),

        LeakyReLU(0.2),

        BatchNormalization(),

        Dense(28 * 28 * 1, activation='tanh'),

        Reshape((28, 28, 1))

    ])

    return model


# Define Discriminator

def build_discriminator():

    model = Sequential([

        Flatten(input_shape=(28, 28, 1)),

        Dense(512),

        LeakyReLU(0.2),

        Dense(256),

        LeakyReLU(0.2),

        Dense(1, activation='sigmoid')

    ])

    return model


# Compile models

generator = build_generator()

discriminator = build_discriminator()

discriminator.compile(

    loss='binary_crossentropy',

    optimizer=Adam(0.0002, 0.5),

    metrics=['accuracy']

)

discriminator.trainable = False


# Build GAN

gan_input = tf.keras.Input(shape=(100,))

gan_output = discriminator(generator(gan_input))

gan = tf.keras.Model(gan_input, gan_output)

gan.compile(

    loss='binary_crossentropy',

    optimizer=Adam(0.0002, 0.5)

)


# Training function

def train_gan(epochs=10000, batch_size=128, sample_interval=1000):

    valid = np.ones((batch_size, 1))

    fake = np.zeros((batch_size, 1))


    for epoch in range(epochs):

        # Train Discriminator

        idx = np.random.randint(0, x_train.shape[0], batch_size)

        real_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))

        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)

        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # Train Generator

        noise = np.random.normal(0, 1, (batch_size, 100))

        g_loss = gan.train_on_batch(noise, valid)


        if epoch % sample_interval == 0:

            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

            sample_images(epoch)


# Function to generate images

def sample_images(epoch, rows=5, cols=5):

    noise = np.random.normal(0, 1, (rows * cols, 100))

    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5


    fig, axs = plt.subplots(rows, cols, figsize=(5, 5))

    count = 0


    for i in range(rows):

        for j in range(cols):

            axs[i, j].imshow(

                generated_images[count, :, :, 0],

                cmap='gray'

            )

            axs[i, j].axis('off')

            count += 1

    plt.show()


# Train the GAN

train_gan(

    epochs=10000,

    batch_size=128,

    sample_interval=1000

)