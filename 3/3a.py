import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, Nadam, Adamax

SEED = 2017

# Load dataset
data = pd.read_csv('Data/winequality-red.csv', sep=';')

# Separate features and target
y = data['quality']
X = data.drop(['quality'], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=SEED
)

# Create model
def create_model(opt):

    model = Sequential()

    model.add(Dense(
        100,
        input_dim=X_train.shape[1],
        activation='relu'
    ))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(25, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='linear'))

    return model


# Create callbacks
def create_callbacks(opt):

    callbacks = [

        EarlyStopping(
            monitor='val_accuracy',
            patience=200,
            verbose=2
        ),

        ModelCheckpoint(
            'optimizers_best_' + opt + '.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )

    ]

    return callbacks


# Different optimizers
opts = {

    'sgd': SGD(),

    'sgd_0001': SGD(
        learning_rate=0.0001
    ),

    'adam': Adam(),

    'adadelta': Adadelta(),

    'rmsprop': RMSprop(),

    'rmsprop_0001': RMSprop(
        learning_rate=0.0001
    ),

    'nadam': Nadam(),

    'adamax': Adamax()

}

batch_size = 128
n_epochs = 100

results = []

# Train model using different optimizers
for opt in opts:

    print("\nRunning Optimizer:", opt)

    model = create_model(opt)

    callbacks = create_callbacks(opt)

    model.compile(
        loss='mse',
        optimizer=opts[opt],
        metrics=['accuracy']
    )

    hist = model.fit(
        X_train.values,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(X_val.values, y_val),
        verbose=1,
        callbacks=callbacks
    )

    best_epoch = np.argmax(hist.history['val_accuracy'])

    best_acc = hist.history['val_accuracy'][best_epoch]

    # Load best model
    best_model = create_model(opt)

    best_model.load_weights(
        'optimizers_best_' + opt + '.keras'
    )

    best_model.compile(
        loss='mse',
        optimizer=opts[opt],
        metrics=['accuracy']
    )

    score = best_model.evaluate(
        X_test.values,
        y_test,
        verbose=0
    )

    results.append([
        opt,
        best_epoch,
        best_acc,
        score[1]
    ])

# Display results
res = pd.DataFrame(results)

res.columns = [
    'optimizer',
    'epochs',
    'val_accuracy',
    'test_accuracy'
]

print(res)