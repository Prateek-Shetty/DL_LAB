#Implement a basic RNN for sequence prediction.

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, Dense


# Generate synthetic sequential data (e.g., a sine wave)

def generate_sequence(n_timesteps):

    x = np.linspace(0, 50, n_timesteps)

    y = np.sin(x)

    return y


# Prepare dataset

n_timesteps = 100

sequence = generate_sequence(n_timesteps)


# Create input-output pairs for training (Sliding window method)

X, y = [], []

seq_length = 10


# Number of previous steps used for prediction

for i in range(len(sequence) - seq_length):

    X.append(sequence[i:i + seq_length])

    y.append(sequence[i + seq_length])


X, y = np.array(X), np.array(y)


# Reshape input for RNN [samples, timesteps, features]

X = X.reshape((X.shape[0], X.shape[1], 1))


# Build RNN model

model = Sequential([

    SimpleRNN(

        10,

        activation='relu',

        return_sequences=False,

        input_shape=(seq_length, 1)

    ),

    Dense(1)

])


# Compile model

model.compile(

    optimizer='adam',

    loss='mse'

)


# Train model

model.fit(

    X,

    y,

    epochs=100,

    verbose=1

)


# Make predictions

predictions = model.predict(X)


# Print expected vs predicted output

print(f"Expected Output: {y[:5]}")

print(f"Predicted Output: {predictions[:5].flatten()}")