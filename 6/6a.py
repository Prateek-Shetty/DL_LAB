import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Load dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# Display image

plt.imshow(X_train[0], cmap="gray")


# Check shapes

print(X_train.shape)
print(X_test.shape)


# View random image

i = random.randint(1, 60000)

plt.imshow(X_train[i], cmap='gray')


# Display label

label = y_train[i]

print(label)


# View images in grid format

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(
    L_grid,
    W_grid,
    figsize=(17, 17)
)

axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, W_grid * L_grid):

    index = np.random.randint(0, n_training)

    axes[i].imshow(X_train[index])

    axes[i].set_title(y_train[index], fontsize=8)

    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# Normalize dataset

X_train = X_train / 255
X_test = X_test / 255


# Add noise to training data

noise_factor = 0.3

noise_dataset = []

for img in X_train:

    noisy_image = img + noise_factor * np.random.randn(*img.shape)

    noisy_image = np.clip(noisy_image, 0., 1.)

    noise_dataset.append(noisy_image)

noise_dataset = np.array(noise_dataset)

print(noise_dataset.shape)


# Display noisy image

plt.imshow(noise_dataset[22], cmap="gray")


# Add noise to test data

noise_test_set = []

for img in X_test:

    noisy_image = img + noise_factor * np.random.randn(*img.shape)

    noisy_image = np.clip(noisy_image, 0., 1.)

    noise_test_set.append(noisy_image)

noise_test_set = np.array(noise_test_set)

print(noise_test_set.shape)


# Build autoencoder model

autoencoder = tf.keras.models.Sequential()


# Encoder

autoencoder.add(

    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=2,
        padding="same",
        input_shape=(28, 28, 1)
    )
)

autoencoder.add(

    tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        strides=2,
        padding="same"
    )
)


# Encoded image

autoencoder.add(

    tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        strides=1,
        padding="same"
    )
)


# Decoder

autoencoder.add(

    tf.keras.layers.Conv2DTranspose(
        filters=16,
        kernel_size=3,
        strides=2,
        padding="same"
    )
)

autoencoder.add(

    tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=2,
        activation='sigmoid',
        padding="same"
    )
)


# Compile model

autoencoder.compile(

    loss='binary_crossentropy',

    optimizer=tf.keras.optimizers.Adam(
        lr=0.001
    )

)

autoencoder.summary()


# Train model

autoencoder.fit(

    noise_dataset.reshape(-1, 28, 28, 1),

    X_train.reshape(-1, 28, 28, 1),

    epochs=10,

    batch_size=200,

    validation_data=(

        noise_test_set.reshape(-1, 28, 28, 1),

        X_test.reshape(-1, 28, 28, 1)

    )

)


# Evaluate model

evaluation = autoencoder.evaluate(

    noise_test_set.reshape(-1, 28, 28, 1),

    X_test.reshape(-1, 28, 28, 1)

)

print('Test Accuracy : {:.3f}'.format(evaluation))


# Predict images

predicted = autoencoder.predict(

    noise_test_set[:10].reshape(-1, 28, 28, 1)

)

print(predicted.shape)


# Display noisy and reconstructed images

fig, axes = plt.subplots(

    nrows=2,

    ncols=10,

    sharex=True,

    sharey=True,

    figsize=(20, 4)

)

for images, row in zip(

    [noise_test_set[:10], predicted],

    axes

):

    for img, ax in zip(images, row):

        ax.imshow(

            img.reshape((28, 28)),

            cmap='Greys_r'

        )

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)