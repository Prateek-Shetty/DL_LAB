import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Function to create model
def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(
            128,
            activation='relu',
            input_shape=(784,)
        ),

        tf.keras.layers.Dense(
            64,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            10,
            activation='softmax'
        )

    ])

    return model


# Different optimizers
optimizers = {

    "SGD": tf.keras.optimizers.SGD(),

    "Adam": tf.keras.optimizers.Adam(),

    "RMSprop": tf.keras.optimizers.RMSprop()

}


# Train model using each optimizer
for name, optimizer in optimizers.items():

    print("\nOptimizer:", name)

    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=5
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test
    )

    print("Test Accuracy:", accuracy)