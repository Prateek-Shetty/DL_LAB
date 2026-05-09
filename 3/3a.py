import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, Nadam, Adamax

SEED = 2017

data = pd.read_csv('Data/winequality-red.csv', sep=';')

y = data['quality']

X = data.drop(['quality'], axis=1)

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


def create_callbacks(opt):

    callbacks = [

        EarlyStopping(
            monitor='val_acc',
            patience=200,
            verbose=2
        ),

        ModelCheckpoint(
            'checkpoints/optimizers_best_' + opt + '.h5',
            monitor='val_acc',
            save_best_only=True,
            verbose=0
        )

    ]

    return callbacks


opts = dict({

    'sgd': SGD(),

    'sgd-0001': SGD(
        lr=0.0001,
        decay=0.00001
    ),

    'adam': Adam(),

    'adadelta': Adadelta(),

    'rmsprop': RMSprop(),

    'rmsprop-0001': RMSprop(
        lr=0.0001
    ),

    'nadam': Nadam(),

    'adamax': Adamax()

})

batch_size = 128

n_epochs = 1000

results = []


# Loop through the optimizers

for opt in opts:

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
        verbose=0,
        callbacks=callbacks
    )

    best_epoch = np.argmax(hist.history['val_acc'])

    best_acc = hist.history['val_acc'][best_epoch]

    best_model = create_model(opt)

    # Load the model weights with the highest validation accuracy

    best_model.load_weights(
        'checkpoints/optimizers_best_' + opt + '.h5'
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


res = pd.DataFrame(results)

res.columns = [
    'optimizer',
    'epochs',
    'val_accuracy',
    'test_accuracy'
]

print(res)