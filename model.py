from keras import Input, Model
from keras.layers import LSTM, Dense, Concatenate, Flatten
from numpy import concatenate
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback
from keras.initializers import RandomNormal

import numpy as np
import random
import io
import os

from tensorflow.python.layers.base import Layer

#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})

modelName = 'noGpt.h5'

"""
## Build the model: a single LSTM layer
"""

saveEveryFit = 10

minChar = ord(' ')
maxChar = ord('~')
nChars = maxChar - minChar

seqLen = 32 # Sequence length for LSTM layer
tokensBag = 256

epochsPerSeq = 20
batchSize = 50

model = None

def generateModel():
    global model
    global seqLen
    global batchSize
    global tokensBag
    global nChars

    # Define the first input
    input_1 = Input(shape=(seqLen, tokensBag))  # Replace additional_input_dim with the actual dimension of your second input
    #flatten_1 = Flatten()(input_1)

    # Define the second input
    input_2 = Input(shape=(seqLen, nChars))
    #flatten_2 = Flatten()(input_2)

    # Define the first input branch
    lstm_1 = LSTM(tokensBag*2, kernel_initializer=RandomNormal(mean=0.0, stddev=1.0))(input_1)

    # Define the second input branch
    lstm_2 = LSTM(tokensBag*2)(input_2)

    # Concatenate the LSTM outputs
    merged_outputs = Concatenate()([lstm_1, lstm_2])

    # Play
    dense1 = Dense(tokensBag*2)(merged_outputs)

    denseOut = dense1

    # Output 1: Dense layer for classification
    output_1 = Dense(tokensBag, activation='relu', name='output_1')(denseOut)

    # Output 2: Another Dense layer for regression
    output_2 = Dense(nChars, activation='softmax', name='output_2')(denseOut)

    # Create the model
    model = Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

    # Compile the model and specify the loss, optimizer, and metrics for each output
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                  loss={'output_1': 'mean_squared_error', 'output_2': 'binary_crossentropy'},
                  metrics={'output_1': 'accuracy', 'output_2': 'accuracy'})

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
        curSeqChars = curSeqChars[1:]

    if (len(curSeqBag) > seqLen):
        curSeqBag = curSeqBag[1:]

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
    x1 = np.reshape(x1, psb_shape)
    x2 = np.reshape(x2, psc_shape)

    #x2 = (x2 >= 0.5).astype(int)

    res = model.predict([x1, x2])
    return res

def print_callback(epoch, logs):
    global epochsPerSeq
    print(f" Epoch {epoch + 1}/{epochsPerSeq}, Loss: {logs['loss']}, Accuracy: {logs['output_1_accuracy']}")

def pad(seq, size):
    global seqLen

    zero = None
    while(len(seq) < seqLen):
        if zero is None:
            zero = np.zeros(size)

        seq.insert(0, zero)

    return seq


def printCharSeq():
    global curSeqChars
    global minChar

    res = ""
    for chars in curSeqChars:
        max = np.argmax(chars)
        res += chr(max+minChar)

    print("curSeq: ", res)

fitNum = 0

def fitSeq():
    global saveEveryFit
    global fitNum

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
        psb = np.reshape(psb, psb_shape)
        psc = np.reshape(psc, psc_shape)

        csb_shape = (1, tokensBag)
        csc_shape = (1, nChars)
        csb = np.reshape(csb, csb_shape)
        csc = np.reshape(csc, csc_shape)

        #psc = (psc >= 0.5).astype(int)
        #csc = (csc >= 0.5).astype(int)

        #print("psb shape: ", psb.shape)
        #print("psc shape: ", psc.shape)
        #print("csb shape: ", csb.shape)
        #print("csc shape: ", csc.shape)

        input = [psb, psc]
        output = [csb, csc]

        #printCharSeq()

        model.fit(input, output, epochs=epochsPerSeq, batch_size=batchSize, callbacks=[LambdaCallback(on_epoch_end=print_callback)])

        if fitNum % saveEveryFit == 0:
            model.save(modelName)

        fitNum += 1

    res = predictSeq()
    prevBag = res[0][-1]

# Default
initBag()