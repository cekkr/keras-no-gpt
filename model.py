from keras import Input, Model
from keras.src.layers import LSTM, Dense, Concatenate, Flatten
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

modelName = 'noGpt.h5'

"""
## Build the model: a single LSTM layer
"""

minChar = ord(' ')
maxChar = ord('~')
nChars = maxChar - minChar

seqLen = 16 # Sequence length for LSTM layer
tokensBag = 256

epochsPerSeq = 10
batchSize = 1

model = None

def generateModel():
    global model
    global seqLen
    global batchSize
    global tokensBag
    global nChars

    # Define the first input
    input_1 = Input(shape=(seqLen, tokensBag))  # Replace additional_input_dim with the actual dimension of your second input
    flatten_1 = Flatten()(input_1)

    # Define the second input
    input_2 = Input(shape=(seqLen, nChars))
    flatten_2 = Flatten()(input_2)


    # Define the first input branch
    lstm_1 = Dense(tokensBag*2)(flatten_1)

    # Define the second input branch
    lstm_2 = Dense(tokensBag*2)(flatten_2)

    # Concatenate the LSTM outputs
    merged_outputs = Concatenate()([lstm_1, lstm_2])

    # Play
    dense1 = Dense(tokensBag*2)(merged_outputs)
    dense2 = Dense(tokensBag*2)(dense1)

    # Output 1: Dense layer for classification
    output_1 = Dense(tokensBag, activation='softmax', name='output_1')(dense2)

    # Output 2: Another Dense layer for regression
    output_2 = Dense(nChars, activation='linear', name='output_2')(dense2)

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
    global tokensBag
    global nChars
    global seqLen

    x1 = curSeqBag[:]
    x2 = curSeqChars[:]

    x1 = pad(x1, tokensBag)
    x2 = pad(x2, nChars)

    x1 = np.array(x1)
    x2 = np.array(x2)

    psb_shape = (1, seqLen, tokensBag)
    psc_shape = (1, seqLen, nChars)
    x1 = np.zeros(psb_shape)
    x2 = np.zeros(psc_shape)

    return model.predict([x1, x2])

def print_callback(epoch, logs):
    global epochsPerSeq

    print(f"Epoch {epoch + 1}/{epochsPerSeq}, Loss: {logs['loss']}, Accuracy: {logs['output_1_accuracy']}")

def pad(seq, size):
    global seqLen

    zero = None
    while(len(seq) < seqLen):
        if zero is None:
            zero = np.zeros(size)

        seq.insert(0, zero)

    return seq


def fitSeq():
    global batchSize
    global epochsPerSeq
    global prevSeqChars
    global prevSeqBag
    global curSeqBag
    global curSeqChars
    global prevBag

    global nChars
    global tokensBag

    if len(prevSeqChars) > 0:

        psb = prevSeqBag[:]
        psc = prevSeqChars[:]

        psb = pad(psb, tokensBag)
        psc = pad(psc, nChars)

        csb = curSeqBag[-1]
        csc = curSeqChars[-1]

        #csb = pad(csb, tokensBag)
        #csc = pad(csc, nChars)

        psb = np.array(psb)
        psc = np.array(psc)
        csb = np.array(csb)
        csc = np.array(csc)

        psb_shape = (1, seqLen, tokensBag)
        psc_shape = (1, seqLen, nChars)
        psb = np.zeros(psb_shape)
        psc = np.zeros(psc_shape)

        csb_shape = (1, tokensBag)
        csc_shape = (1, nChars)
        csb = np.zeros(csb_shape)
        csc = np.zeros(csc_shape)

        print("psb shape: ", psb.shape)
        print("psc shape: ", psc.shape)
        print("csb shape: ", csb.shape)
        print("csc shape: ", csc.shape)

        input = [psb, psc]
        output = [csb, csc]

        model.fit(input, output, epochs=epochsPerSeq, batch_size=batchSize, callbacks=[LambdaCallback(on_epoch_end=print_callback)])

        model.save(modelName)

        res = predictSeq()
        prevBag = res[0][-1]

# Default
initBag()