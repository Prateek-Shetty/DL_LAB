#Fine-tune a pretrained model like ResNet50 or EfficientNet on a custom dataset

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

import os


# Define dataset directories

data_dir = "path/to/dataset"   # Change this to your dataset path

train_dir = os.path.join(data_dir, "train")

val_dir = os.path.join(data_dir, "val")


# Define parameters

img_size = (224, 224)

batch_size = 32

num_classes = len(os.listdir(train_dir))

epochs = 10


# Data Augmentation and Preprocessing

datagen_train = ImageDataGenerator(

    rescale=1.0/255,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    validation_split=0.2

)

datagen_val = ImageDataGenerator(

    rescale=1.0/255

)


train_generator = datagen_train.flow_from_directory(

    train_dir,

    target_size=img_size,

    batch_size=batch_size,

    class_mode='categorical'

)


val_generator = datagen_val.flow_from_directory(

    val_dir,

    target_size=img_size,

    batch_size=batch_size,

    class_mode='categorical'

)


# Load Pretrained Model

base_model = ResNet50(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)

)

base_model.trainable = False


# Add Custom Layers

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

out = Dense(num_classes, activation='softmax')(x)


# Compile Model

model = Model(

    inputs=base_model.input,

    outputs=out

)

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# Train Model

model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=epochs,

    steps_per_epoch=train_generator.samples // batch_size,

    validation_steps=val_generator.samples // batch_size

)


# Fine-tune the model by unfreezing some layers

base_model.trainable = True


for layer in base_model.layers[:100]:

    layer.trainable = False


# Compile again with lower learning rate

model.compile(

    optimizer=Adam(learning_rate=0.0001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# Train again for fine-tuning

model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=epochs // 2,

    steps_per_epoch=train_generator.samples // batch_size,

    validation_steps=val_generator.samples // batch_size

)


# Save Model

model.save("fine_tuned_resnet50.h5")