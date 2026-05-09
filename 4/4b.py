import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# Load dataset

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(

    "dataset/train",

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical'

)

val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(

    "dataset/val",

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical'

)


# Load pretrained ResNet50 model

base_model = ResNet50(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)

)

# Freeze pretrained layers

base_model.trainable = False


# Create model

model = Sequential([

    base_model,

    Flatten(),

    Dense(128, activation='relu'),

    Dense(train_data.num_classes, activation='softmax')

])


# Compile model

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# Train model

model.fit(

    train_data,

    validation_data=val_data,

    epochs=5

)


# Fine tuning

base_model.trainable = True


# Compile again

model.compile(

    optimizer=Adam(learning_rate=0.0001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# Train again

model.fit(

    train_data,

    validation_data=val_data,

    epochs=3

)


# Save model

model.save("resnet50_model.h5")