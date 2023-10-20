from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Embedding
from keras.optimizers import Adam, RMSprop

# Set up the crypto parameters: message, key, and ciphertext bit lengths
m_bits = 16  # message
puk_bits = 1424  # public key
prk_bits = 1816  # private key
c_bits = (m_bits+puk_bits)//2  # ciphertext
pad = 'same'

# Compute the size of the message space, used later in training
m_train = 2**(m_bits)
# m_train = 1024000

# Alice network
ainput0 = Input(shape=(m_bits))  # message
ainput1 = Input(shape=(puk_bits,))  # public key

ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + puk_bits), activation='tanh')(ainput)
areshape = Reshape((m_bits + puk_bits, 1,))(adense1)

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


# Bob network
binput0 = Input(shape=(c_bits,))  # ciphertext
binput1 = Input(shape=(prk_bits,))  # private key

binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=(m_bits*2), activation='tanh')(binput)
breshape = Reshape((m_bits*2, 1,))(bdense1)

bconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                padding=pad, activation='tanh')(breshape)
bconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                padding=pad, activation='tanh')(bconv1)
bconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                padding=pad, activation='tanh')(bconv2)
bconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                padding=pad, activation='sigmoid')(bconv3)

boutput = Flatten()(bconv4)


bob = Model(inputs=[binput0, binput1],
            outputs=boutput, name='bob')

# # Eve network
einput = Input(shape=(c_bits,))  # ciphertext only

edense1 = Dense(units=(m_bits*2), activation='tanh')(einput)
edense2 = Dense(units=(m_bits*2), activation='tanh')(edense1)
ereshape = Reshape((m_bits*2, 1,))(edense2)

econv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                padding=pad, activation='tanh')(ereshape)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                padding=pad, activation='tanh')(econv1)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                padding=pad, activation='tanh')(econv2)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                padding=pad, activation='sigmoid')(econv3)

eoutput = Flatten()(econv4)  # Eve's attempt at guessing the plaintext

eve = Model(einput, eoutput, name='eve')

# Loss and optimizer
aliceout = alice([ainput0, ainput1])
bobout = bob([aliceout, binput1])  # bob sees ciphertext AND key
eveout = eve(aliceout)  # eve doesn't see the key [aliceout, ainput1]

eveloss = K.mean(K.sum(K.abs(ainput0 - eveout), axis=-1))
bobloss = K.mean(K.sum(K.abs(ainput0 - bobout), axis=-1))

abeloss = bobloss + K.square(m_bits/2 - eveloss) / \
    ((m_bits//2)**2)  # alice-bob loss

# Build and compile the ABE model, used for training Alice-Bob networks
#
abemodel = Model([ainput0, ainput1, binput1],
                 bobout, name='abemodel')
abemodel.add_loss(abeloss)


# alice.compile(loss='mse', optimizer='sgd')
# beoptim = RMSprop(lr=0.001)
# eveoptim = RMSprop(lr=0.001)
abemodel.compile(optimizer='RMSprop')

# Build and compile the Eve model, used for training Eve net (with Alice frozen)
#
alice.trainable = False
evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
evemodel.add_loss(eveloss)
evemodel.compile(optimizer='RMSprop')
