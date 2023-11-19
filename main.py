from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

"""
## Build the model: a single LSTM layer
"""
maxlen = 40


model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        layers.Dense(len(chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
