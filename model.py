from keras import Input, Model
from keras.src.layers import LSTM, Dense, Concatenate
from numpy import concatenate
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import random
import io
import os

from tensorflow.python.layers.base import Layer

modelName = 'noSql.h5'

"""
## Build the model: a single LSTM layer
"""

seqLen = 16 # Sequence length for LSTM layer

minChar = ord(' ')
maxChar = ord('~')
nChars = maxChar - minChar

tokensBag = 256

epochsPerSeq = 10

model = None

def generateModel():
    global model

    # Define the first input
    input_1 = Input(shape=(seqLen, nChars))

    # Define the second input
    input_2 = Input(shape=(seqLen, tokensBag))  # Replace additional_input_dim with the actual dimension of your second input

    # Define the first input branch
    lstm_1 = LSTM(tokensBag)(input_1)

    # Define the second input branch
    lstm_2 = LSTM(tokensBag)(input_2)

    # Concatenate the LSTM outputs
    merged_outputs = Concatenate()([lstm_1, lstm_2])

    # Output 1: Dense layer for classification
    output_1 = Dense(tokensBag, activation='softmax', name='output_1')(merged_outputs)

    # Output 2: Another Dense layer for regression
    output_2 = Dense(nChars, activation='linear', name='output_2')(merged_outputs)

    # Create the model
    model = Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

    # Compile the model and specify the loss, optimizer, and metrics for each output
    optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss={'output_1': 'categorical_crossentropy', 'output_2': 'mean_squared_error'},
                  metrics={'output_1': 'accuracy', 'output_2': 'mae'})

if os.path.exists(modelName):
    # Load the saved model
    model = load_model(modelName)
    print("Model loaded successfully.")
else:
    generateModel()

# Print a summary of the model architecture
model.summary()

###
###

curSeqChars = []
curSeqBag = []

prevSeqChars = []
prevSeqBag = []

prevBag = None

def initBag():
    global prevBag
    global prevSeqChars
    global prevSeqBag

    prevBag = np.zeros(tokensBag)
    prevSeqChars = prevSeqBag = []

def pushChar(ch):
    global prevSeqChars
    global prevSeqBag
    global curSeqChars
    global curSeqBag

    chNum = ord(ch)
    if chNum < minChar or chNum > maxChar:
        #print("char out of bounds")
        return

    chNum -= minChar

    x_pred = np.zeros(nChars)
    x_pred[chNum] = 1

    prevSeqChars = curSeqChars[:]
    prevSeqBag = curSeqBag[:]

    curSeqChars.append(x_pred)
    curSeqBag.append(prevBag)

    if(len(curSeqChars) > seqLen):
        curSeqChars.pop()

    if (len(curSeqBag) > seqLen):
        curSeqBag.pop()

    fitSeq()

def predictSeq():
    global curSeqBag
    global curSeqChars

    return model.predict([curSeqBag, curSeqChars])

def print_callback(epoch, logs):
    global epochsPerSeq

    print(f"Epoch {epoch + 1}/{epochsPerSeq}, Loss: {logs['loss']}, Accuracy: {logs['output_1_accuracy']}")

def fitSeq():
    global epochsPerSeq
    global prevSeqChars
    global prevSeqBag
    global curSeqBag
    global curSeqChars
    global prevBag

    if len(prevSeqChars) > 0:
        model.fit([prevSeqChars, prevSeqBag], [curSeqBag, curSeqChars], epochs=epochsPerSeq, batch_size=1, callbacks=[LambdaCallback(on_epoch_end=print_callback)])
        model.save(modelName)

        res = predictSeq()
        prevBag = res[0]

# Default
initBag()