import tensorflow as tf

RECURRENT_INITIALISER = 'glorot_uniform'


class Encoder(tf.keras.Model):
    dimension = 256

    def __init__(self, vocabulary, dimension, encdoding, batchSize):
        super(Encoder, self).__init__()
        self.encdoding = encdoding
        # GRU - Grated Recurrent Unit is an RNN
        self.gru = tf.keras.layers.GRU(encdoding,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer=RECURRENT_INITIALISER)
        # 'glorot_uniform' draws samples from a uniform distribution
        self.batchSize = batchSize
        # Defining the word embedding to be fed into the neural network (GRU)
        self.embedding = tf.keras.layers.Embedding(vocabulary, 256)

    def call(self, x, hiddenState):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hiddenState)
        return output, state

    def hiddenStateInit(self):
        return tf.zeros((self.batchSize, self.encdoding))


# Inspired and modified from TensorFlow example
# TensorFlow Addons Networks : Sequence-to-Sequence NMT with Attention Mechanism
# 2021
