#Explore a pretrained model (e.g., MobileNet) on a transfer learning task.


# Imports required packages

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Loads fashion mnist dataset

fashion = tf.keras.datasets.fashion_mnist.load_data()


# Each training and test example is assigned to one of the following labels

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# Considering dataset is organized in tuple

(X_train_full, y_train_full), (X_test, y_test) = fashion


# Checks the shape of the datasets

print("Train dataset shape:", X_train_full.shape)
print("Test dataset shape:", X_test.shape)


# Checks the data type

print(X_train_full.dtype)


# Normalize the data

X_train_full, X_test = X_train_full / 255., X_test / 255.


# Finds the index for Pullover and T-shirt/top

class_0_index = class_names.index("Pullover")
class_1_index = class_names.index("T-shirt/top")

print("Index of class_0:", class_0_index)
print("Index of class_1:", class_1_index)


# Gets indexes containing either class

class_0_1_index_flag = [

    True if (x == class_0_index or x == class_1_index)
    else False

    for x in y_train_full
]

print(class_0_1_index_flag[:10])


# Separates dataset containing two classes

X_train_2_classes_full = X_train_full[class_0_1_index_flag]

print(X_train_2_classes_full.shape)


# Flips bool values

class_0_1_index_flag_flipped = [

    not flag for flag in class_0_1_index_flag
]

print(class_0_1_index_flag_flipped[:10])


# Separates dataset containing remaining 8 classes

X_train_8_classes_full = X_train_full[
    class_0_1_index_flag_flipped
]

print(X_train_8_classes_full.shape)


# Separates targets

y_train_2_classes_full = y_train_full[
    class_0_1_index_flag
]

y_train_8_classes_full = y_train_full[
    class_0_1_index_flag_flipped
]

print(y_train_2_classes_full.shape)
print(y_train_8_classes_full.shape)


# Separates validation dataset

X_train_8_classes, X_val_8_classes, y_train_8_classes, y_val_8_classes = train_test_split(

    X_train_8_classes_full,
    y_train_8_classes_full,

    test_size=5000,

    random_state=42,

    stratify=y_train_8_classes_full
)


# Prints shapes

print(X_train_8_classes.shape)
print(X_val_8_classes.shape)


# Standardizes datasets

pixel_means_8_classes = X_train_8_classes.mean(
    axis=0,
    keepdims=True
)

pixel_stds_8_classes = X_train_8_classes.std(
    axis=0,
    keepdims=True
)

X_train_8_classes_scaled = (

    X_train_8_classes - pixel_means_8_classes

) / pixel_stds_8_classes


X_val_8_classes_scaled = (

    X_val_8_classes - pixel_means_8_classes

) / pixel_stds_8_classes


# Encodes labels

label_encoder_8_classes = LabelEncoder()

y_train_8_classes_encoded = label_encoder_8_classes.fit_transform(
    y_train_8_classes
)

y_val_8_classes_encoded = label_encoder_8_classes.transform(
    y_val_8_classes
)


# Creates model

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=[28, 28]),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        8,
        activation="softmax"
    )

])


# Compiles model

model.compile(

    loss="sparse_categorical_crossentropy",

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001
    ),

    metrics=["accuracy"]

)


# Model summary

model.summary()


# Fits the model

model_history = model.fit(

    X_train_8_classes_scaled,

    y_train_8_classes_encoded,

    epochs=20,

    validation_data=(
        X_val_8_classes_scaled,
        y_val_8_classes_encoded
    )

)


# Saves the model

model.save("./models/my_fashion_mnist_model.keras")


# Separates validation dataset for 2 classes

X_train_2_classes, X_val_2_classes, y_train_2_classes, y_val_2_classes = train_test_split(

    X_train_2_classes_full,

    y_train_2_classes_full,

    test_size=3000,

    random_state=42,

    stratify=y_train_2_classes_full
)


# Prints shape

print(X_train_2_classes.shape)
print(X_val_2_classes.shape)


# Standardizes datasets

pixel_means_2_classes = X_train_2_classes.mean(
    axis=0,
    keepdims=True
)

pixel_stds_2_classes = X_train_2_classes.std(
    axis=0,
    keepdims=True
)

X_train_2_classes_scaled = (

    X_train_2_classes - pixel_means_2_classes

) / pixel_stds_2_classes


X_val_2_classes_scaled = (

    X_val_2_classes - pixel_means_2_classes

) / pixel_stds_2_classes


# Encodes labels

label_encoder_2_classes = LabelEncoder()

y_train_2_classes_encoded = label_encoder_2_classes.fit_transform(
    y_train_2_classes
)

y_val_2_classes_encoded = label_encoder_2_classes.transform(
    y_val_2_classes
)


# Clears session

tf.keras.backend.clear_session()

tf.random.set_seed(42)


# Creates model from scratch

model_from_scratch = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=[28, 28]),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])


# Compiles model

model_from_scratch.compile(

    loss="binary_crossentropy",

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001
    ),

    metrics=["accuracy"]

)


# Model summary

model_from_scratch.summary()


# Fits model

model_from_scratch_history = model_from_scratch.fit(

    X_train_2_classes_scaled,

    y_train_2_classes_encoded,

    epochs=20,

    validation_data=(
        X_val_2_classes_scaled,
        y_val_2_classes_encoded
    )

)


# Gets test indexes

class_0_1_index_flag = [

    True if (x == class_0_index or x == class_1_index)
    else False

    for x in y_test
]


# Separates test dataset

X_test_2_classes = X_test[class_0_1_index_flag]

print(X_test_2_classes.shape)


# Separates targets

y_test_2_classes = y_test[class_0_1_index_flag]


# Encodes labels

y_test_2_classes_encoded = label_encoder_2_classes.transform(
    y_test_2_classes
)

print(y_test_2_classes_encoded)


# Standardizes test set

X_test_2_classes_scaled = (

    X_test_2_classes - pixel_means_2_classes

) / pixel_stds_2_classes


# Evaluates model

model_from_scratch.evaluate(

    X_test_2_classes_scaled,

    y_test_2_classes_encoded

)


# Loads pretrained model

model_using_pretrained_layers = tf.keras.models.load_model(

    "./models/my_fashion_mnist_model.keras"

)


# Model summary

model_using_pretrained_layers.summary()


# Removes last layer

model_using_pretrained_layers.pop()


# Adds binary output layer

model_using_pretrained_layers.add(

    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )

)


# Model summary

model_using_pretrained_layers.summary()


# Uses only 60% of training set

X_train_2_classes_scaled_subset, _, y_train_2_classes_encoded_subset, _ = train_test_split(

    X_train_2_classes_scaled,

    y_train_2_classes_encoded,

    train_size=0.60,

    stratify=y_train_2_classes_encoded
)


# Freezes pretrained layers

for layer in model_using_pretrained_layers.layers[:-1]:

    layer.trainable = False


# Clears session

tf.keras.backend.clear_session()

tf.random.set_seed(42)


# Compiles model

model_using_pretrained_layers.compile(

    loss="binary_crossentropy",

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001
    )

)


# Trains output layer

model_using_pretrained_layers_history = model_using_pretrained_layers.fit(

    X_train_2_classes_scaled_subset,

    y_train_2_classes_encoded_subset,

    epochs=5,

    validation_data=(
        X_val_2_classes_scaled,
        y_val_2_classes_encoded
    )

)


# Makes pretrained layers trainable

for layer in model_using_pretrained_layers.layers[:-1]:

    layer.trainable = True


# Compiles again

model_using_pretrained_layers.compile(

    loss="binary_crossentropy",

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001
    )

)


# Trains again

model_using_pretrained_layers_history = model_using_pretrained_layers.fit(

    X_train_2_classes_scaled_subset,

    y_train_2_classes_encoded_subset,

    epochs=100,

    validation_data=(
        X_val_2_classes_scaled,
        y_val_2_classes_encoded
    )

)


# Evaluates pretrained model

model_using_pretrained_layers.evaluate(

    X_test_2_classes_scaled,

    y_test_2_classes_encoded

)