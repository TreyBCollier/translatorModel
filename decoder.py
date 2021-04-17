import tensorflow as tf
from bahdanau import Bahdanau

RECURRENT_INITIALISER = 'glorot_uniform'


class Decoder(tf.keras.Model):
    dimension = 256

    def __init__(self, vocabulary, dimension, decoding, batchSize):
        super(Decoder, self).__init__()
        self.decoding = decoding
        # GRU - Grated Recurrent Unit is an RNN
        self.gru = tf.keras.layers.GRU(self.decoding,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer=RECURRENT_INITIALISER)
        self.fc = tf.keras.layers.Dense(vocabulary)
        self.batchSize = batchSize
        self.embedding = tf.keras.layers.Embedding(vocabulary, 256)

        # Bahdanau provides attention for decoder
        self.attention = Bahdanau(self.decoding)

    def call(self, x, hidden, enc_output):
        decodeVector, bahdanauWeights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(decodeVector, 1), x], axis=-1)
        output, stateVector = self.gru(x)
        shape = output.shape[2]
        output = tf.reshape(output, (-1, shape))
        x = self.fc(output)

        return x, stateVector, bahdanauWeights

# Inspired and modified from TensorFlow example
# TensorFlow Addons Networks : Sequence-to-Sequence NMT with Attention Mechanism
# 2021