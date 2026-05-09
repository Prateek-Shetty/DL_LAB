import numpy as np
import tensorflow as tf

# Create sequence data
x = np.arange(0, 100)
y = np.sin(x)

# Prepare dataset
X = []
target = []

seq_length = 5

for i in range(len(y) - seq_length):

    X.append(y[i:i + seq_length])

    target.append(y[i + seq_length])

X = np.array(X)
target = np.array(target)

# Reshape data for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build RNN model
model = tf.keras.Sequential([

    tf.keras.layers.SimpleRNN(
        10,
        activation='relu',
        input_shape=(seq_length, 1)
    ),

    tf.keras.layers.Dense(1)

])

# Compile model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train model
model.fit(
    X,
    target,
    epochs=20
)

# Predict output
predictions = model.predict(X)

# Display outputs
print("Actual Values:", target[:5])
print("Predicted Values:", predictions[:5].flatten())