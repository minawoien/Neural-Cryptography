from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate


class NeuralCryptoBuilder:
    def __init__(self, m_bits=16, k_bits=16, c_bits=16, pad='same'):
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits
        self.pad = pad

    def build_input_layers(self, input_shape):
        return Input(shape=input_shape), Input(shape=(self.k_bits))

    def build_dense_layer(self, input_layer, units, activation='tanh'):
        return Dense(units=units, activation=activation)(input_layer)

    def build_conv_layers(self, reshaped_layer, filters, kernel_size, strides, activation='tanh'):
        layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding=self.pad, activation=activation)(reshaped_layer)
        return layer

    def build_reshape_layer(self, layer, shape):
        return Reshape(shape)(layer)

    def build_alice_or_bob(self, name):
        input0, input1 = self.build_input_layers((self.m_bits,))
        input = concatenate([input0, input1], axis=1)

        dense = self.build_dense_layer(input, self.m_bits + self.k_bits)
        reshape = self.build_reshape_layer(
            dense, (self.m_bits + self.k_bits, 1))

        return self.create_model(reshape, [input0, input1], name)

    def build_eve(self, name):
        einput = Input(shape=self.c_bits)  # ciphertext only
        edense1 = self.build_dense_layer(einput, self.c_bits + self.k_bits)
        edense2 = self.build_dense_layer(edense1, self.c_bits + self.k_bits)
        ereshape = self.build_reshape_layer(
            edense2, (self.c_bits + self.k_bits, 1))

        return self.create_model(ereshape, einput, name)

    def create_model(self, reshape, inputs, name):
        conv1 = self.build_conv_layers(reshape, 2, 4, 1)
        conv2 = self.build_conv_layers(conv1, 4, 2, 2)
        conv3 = self.build_conv_layers(conv2, 4, 1, 1)
        conv4 = self.build_conv_layers(conv3, 1, 1, 1, activation='sigmoid')

        output = Flatten()(conv4)

        return Model(inputs=inputs, outputs=output, name=name)

    def compile_model(self, inputs, output, loss, optimizer='RMSprop', name=None):
        model = Model(inputs=inputs, outputs=output, name=name)
        model.add_loss(loss)
        model.compile(optimizer=optimizer)
        return model

    def compute_loss(self, actual, predicted):
        return K.mean(K.sum(K.abs(actual - predicted), axis=-1))

    def compile_abe_model(self, alice, bob, eve, ainput0, ainput1, binput1):
        aliceout = alice([ainput0, ainput1])
        bobout = bob([aliceout, binput1])
        eveout = eve(aliceout)

        bobloss = self.compute_loss(ainput0, bobout)
        eveloss = self.compute_loss(ainput0, eveout)

        abeloss = bobloss + \
            K.square(self.m_bits / 2 - eveloss) / ((self.m_bits // 2) ** 2)

        return self.compile_model([ainput0, ainput1, binput1], bobout, abeloss, name='abemodel')

    def compile_eve_model(self, alice, eve, ainput0, ainput1):
        alice.trainable = False
        aliceout = alice([ainput0, ainput1])
        eveout = eve(aliceout)
        eveloss = self.compute_loss(ainput0, eveout)

        return self.compile_model([ainput0, ainput1], eveout, eveloss, name='evemodel')


m_bits = 16
k_bits = 16
c_bits = 16
m_train = 2**(m_bits)

builder = NeuralCryptoBuilder()

# Build Alice, Bob, and Eve models
alice = builder.build_alice_or_bob('alice')
bob = builder.build_alice_or_bob('bob')
eve = builder.build_eve('eve')

ainput0, ainput1 = Input(shape=(16,)), Input(shape=(16,))
binput1 = Input(shape=(16,))

# Compile the ABE and Eve models
abemodel = builder.compile_abe_model(
    alice, bob, eve, ainput0, ainput1, binput1)
evemodel = builder.compile_eve_model(alice, eve, ainput0, ainput1)
