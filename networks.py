import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Activation, Dense
from tensorflow.python.keras.layers import Reshape, Flatten
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.merge import concatenate

# Set up the crypto parameters: message, key, and ciphertext bit lengths
m_bits = 16
k_bits = 16
c_bits = 16
pad = 'same'

# Compute the size of the message space, used later in training
m_train = 2**(m_bits)  # + k_bits)

# Alice network
ainput0 = Input(shape=(m_bits))  # message
ainput1 = Input(shape=(k_bits))  # key
ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + k_bits), activation='tanh')(ainput)
areshape = Reshape((m_bits + k_bits, 1,))(adense1)

aconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                padding=pad, activation='tanh')(areshape)

aconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                padding=pad, activation='tanh')(aconv1)

aconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                padding=pad, activation='tanh')(aconv2)

aconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                padding=pad, activation='sigmoid')(aconv3)

aoutput = Flatten()(aconv4)

alice = Model(inputs=[ainput0, ainput1],
              outputs=aoutput, name='alice')
