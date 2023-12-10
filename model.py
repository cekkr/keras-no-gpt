from keras import Input, Model
from keras.layers import LSTM, Dense, Concatenate, Flatten
from numpy import concatenate
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback, Callback
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
    lstm_1 = LSTM(tokensBag*4, kernel_initializer=RandomNormal(mean=0.0, stddev=1.0))(input_1)

    # Define the second input branch
    lstm_2 = LSTM(tokensBag*4)(input_2)

    # Concatenate the LSTM outputs
    merged_outputs = Concatenate()([lstm_1, lstm_2])

    # Play
    dense1 = Dense(tokensBag * 4)(merged_outputs)
    dense2 = Dense(tokensBag * 4)(dense1)

    denseOut = dense2

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
###

def pad(seq, size):
    global seqLen

    zero = None
    while(len(seq) < seqLen):
        if zero is None:
            zero = np.zeros(size)

        seq.insert(0, zero)

    return seq

class Batch:
    def __init__(self):
        self.curOps = []

    def addOp(self, op):
        self.curOps.append(op)

    def remOp(self, op):
        self.curOps.remove(op)

    def remOps(self, ops):
        for op in ops:
            self.remOp(op)

    def size(self):
        return len(self.curOps)

    def getTrain(self):
        global tokensBag
        global nChars

        ap1 = []
        ap2 = []

        p1 = []
        p2 = []

        c1 = []
        c2 = []

        finished = []

        for op in self.curOps:
            op.pushChar()

            if op.finish():
                finished.append(op)

            psb = op.prevSeqBag[:]
            psc = op.prevSeqChars[:]

            psb = pad(psb, tokensBag)
            psc = pad(psc, nChars)

            ap1.append(psb)
            ap2.append(psc)

            if len(op.prevSeqChars) > 0: # is not the first cycle
                csb = op.curSeqBag[-1]
                csc = op.curSeqChars[-1]

                p1.append(psb)
                p2.append(psc)

                c1.append(csb)
                c2.append(csc)

        self.remOps(finished)

        return [[p1, p2], [c1, c2], [ap1, ap2]]

    def getPredict(self):
        global tokensBag
        global nChars

        ap1 = []
        ap2 = []

        finished = []

        for op in self.curOps:
            op.pushChar()

            if op.finish():
                finished.append(op)

            psb = op.prevSeqBag[:]
            psc = op.prevSeqChars[:]

            psb = pad(psb, tokensBag)
            psc = pad(psc, nChars)

            ap1.append(psb)
            ap2.append(psc)

        self.remOps(finished)

        return [ap1, ap2]

    class Operation:
        def __init__(self, content):
            global tokensBag

            self.content = content
            self.pos = 0

            self.prevBag = np.zeros(tokensBag)

            self.prevSeqChars = []
            self.prevSeqBag = []

            self.curSeqChars = []
            self.curSeqBag = []

        def next(self):
            self.pos += 1

        def finish(self):
            return self.pos >= len(self.content)

        def pushChar(self):
            global minChar
            global maxChar
            global seqLen
            global nChars

            ch = self.content[self.pos]
            chNum = ord(ch)
            if chNum < minChar or chNum > maxChar:
                # print("char out of bounds")
                return

            chNum -= minChar

            x_pred = np.zeros(nChars)
            x_pred[chNum] = 1

            self.prevSeqChars = self.curSeqChars[:]
            self.prevSeqBag = self.curSeqBag[:]

            self.curSeqChars.append(x_pred)
            self.curSeqBag.append(self.prevBag)

            if len(self.curSeqChars) > seqLen:
                self.curSeqChars = self.curSeqChars[1:]

            if len(self.curSeqBag) > seqLen:
                self.curSeqBag = self.curSeqBag[1:]

            self.next()


batch = Batch()

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

def predictSeq(seq):
    global tokensBag
    global nChars
    global seqLen

    x1 = seq[0]
    x2 = seq[1]

    for i in range(0, len(x1)):
        x1[i] = pad(x1[i], tokensBag)
        x2[i] = pad(x2[i], nChars)

    x1 = np.array(x1)
    x2 = np.array(x2)

    nseq = len(x1)

    psb_shape = (nseq, seqLen, tokensBag)
    psc_shape = (nseq, seqLen, nChars)
    x1 = np.reshape(x1, psb_shape)
    x2 = np.reshape(x2, psc_shape)

    res = model.predict([x1, x2])
    return res


def printCharSeq():
    global curSeqChars
    global minChar

    res = ""
    for chars in curSeqChars:
        max = np.argmax(chars)
        res += chr(max+minChar)

    print("curSeq: ", res)

def print_callback(epoch, logs):
    global epochsPerSeq
    print(f" Epoch {epoch + 1}/{epochsPerSeq}, Loss: {logs['loss']}, Accuracy: {logs['output_1_accuracy']}")

class BatchMetricsCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        print(f'Batch {batch + 1} - Loss: {logs["loss"]:.4f}, Accuracy: {logs["output_1_accuracy"]:.4f}')


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

    global batch

    train = batch.getTrain()

    x = train[0]
    y = train[1]

    nbatches = len(x[0])

    if nbatches > 0:
        x[0] = np.array(x[0])
        x[1] = np.array(x[1])
        y[0] = np.array(y[0])
        y[1] = np.array(y[1])

        psb_shape = (nbatches, seqLen, tokensBag)
        psc_shape = (nbatches, seqLen, nChars)
        psb = np.reshape(x[0], psb_shape)
        psc = np.reshape(x[1], psc_shape)

        csb_shape = (nbatches, tokensBag)
        csc_shape = (nbatches, nChars)
        csb = np.reshape(y[0], csb_shape)
        csc = np.reshape(y[1], csc_shape)

        input = [psb, psc]
        output = [csb, csc]

        #printCharSeq()

        # Create an instance of the custom callback
        batch_metrics_callback = BatchMetricsCallback()

        model.fit(input, output, epochs=epochsPerSeq, batch_size=batchSize, callbacks=[batch_metrics_callback])

        if fitNum % saveEveryFit == 0:
            tf.keras.backend.clear_session()
            model.save(modelName)

        fitNum += 1

    res = predictSeq(train[2])

    for x in range(0, len(res[0])):
        batch.curOps[x].prevBag = res[0][x]

# Default
initBag()