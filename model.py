from keras import Input, Model
from keras.src.layers import LSTM, Dense, Concatenate
from numpy import concatenate
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import random
import io

from tensorflow.python.layers.base import Layer

"""
## Build the model: a single LSTM layer
"""

seqLen = 16 # Sequence length for LSTM layer

minChar = ord(' ')
maxChar = ord('~')
nChars = maxChar - minChar

tokensBag = 256

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

# Print a summary of the model architecture
model.summary()

curSeq = []

def pushChar(ch):
    x_pred = np.zeros(len(nChars))
    x_pred[ord(ch)] = 1

    curSeq.append(x_pred)

    if(len(curSeq) > seqLen):
        curSeq.pop()