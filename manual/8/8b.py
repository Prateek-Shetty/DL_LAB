import tensorflow as tf
import numpy as np

# Sample text
text = "deep learning is powerful and useful"

# Convert characters to unique values
chars = sorted(set(text))

char_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_char = {i:c for i, c in enumerate(chars)}

# Prepare sequence data
seq_length = 5

X = []
y = []

for i in range(len(text) - seq_length):

    seq = text[i:i + seq_length]

    target = text[i + seq_length]

    X.append([char_to_idx[c] for c in seq])

    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)

# Reshape data
X = X.reshape(X.shape[0], X.shape[1], 1) / len(chars)

# Build LSTM model
model = tf.keras.Sequential([

    tf.keras.layers.LSTM(
        64,
        input_shape=(seq_length, 1)
    ),

    tf.keras.layers.Dense(
        len(chars),
        activation='softmax'
    )

])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Train model
model.fit(
    X,
    y,
    epochs=50
)

# Predict next character
test = "deep "

x_input = np.array(
    [[char_to_idx[c] for c in test]]
)

x_input = x_input.reshape(1, seq_length, 1) / len(chars)

prediction = model.predict(x_input)

predicted_char = idx_to_char[np.argmax(prediction)]

print("Predicted Character:", predicted_char)