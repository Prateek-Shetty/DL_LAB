import tensorflow as tf

import numpy as np

import string


# Load text data

text = open("shakespeare.txt", "r").read().lower()

chars = sorted(set(text))


# Map characters to indices

char_to_idx = {

    c: i for i, c in enumerate(chars)

}

idx_to_char = {

    i: c for i, c in enumerate(chars)

}


# Convert text to sequence of numbers

seq_length = 100

sequences = []

next_chars = []


for i in range(len(text) - seq_length):

    sequences.append([

        char_to_idx[c]

        for c in text[i:i + seq_length]

    ])

    next_chars.append(

        char_to_idx[text[i + seq_length]]

    )


X = np.array(sequences)

y = np.array(next_chars)


# Reshape input for LSTM

X = X.reshape(

    X.shape[0],

    X.shape[1],

    1

) / len(chars)


# Build LSTM model

model = tf.keras.Sequential([

    tf.keras.layers.LSTM(

        128,

        input_shape=(seq_length, 1),

        return_sequences=True

    ),

    tf.keras.layers.LSTM(128),

    tf.keras.layers.Dense(

        len(chars),

        activation="softmax"

    )

])


# Compile model

model.compile(

    loss="sparse_categorical_crossentropy",

    optimizer="adam"

)


# Train model

model.fit(

    X,

    y,

    epochs=20,

    batch_size=64

)


# Function to generate text

def generate_text(seed_text, length=200):

    generated = seed_text

    for _ in range(length):

        x_input = np.array([

            [

                char_to_idx[c]

                for c in generated[-seq_length:]

            ]

        ]) / len(chars)

        x_input = x_input.reshape(

            1,

            seq_length,

            1

        )

        predicted_idx = np.argmax(

            model.predict(x_input)

        )

        generated += idx_to_char[predicted_idx]

    return generated


# Generate new text

print(

    generate_text(

        "shall i compare thee to a summer's day? "

    )

)