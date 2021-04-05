import tensorflow as tf
from bahdanau import Bahdanau

RECURRENT_INITIALISER = 'glorot_uniform'


class Decoder(tf.keras.Model):
    dimension = 256

    def __init__(self, vocabulary, dimension, decoding, batchSize):
        super(Decoder, self).__init__()
        self.decoding = decoding
        self.gru = tf.keras.layers.GRU(self.decoding,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer=RECURRENT_INITIALISER)
        self.fc = tf.keras.layers.Dense(vocabulary)
        self.batchSize = batchSize
        self.embedding = tf.keras.layers.Embedding(vocabulary, 256)

        # used for attention
        self.attention = Bahdanau(self.decoding)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (64, max_length, hidden_size)
        decodeVector, bahdanauWeights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (64, 1, 256)
        x = self.embedding(x)

        # x shape after concatenation == (64, 1, 256 + hidden_size)
        x = tf.concat([tf.expand_dims(decodeVector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, stateVector = self.gru(x)

        # output shape == (64 * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (64, vocab)
        x = self.fc(output)

        return x, stateVector, bahdanauWeights
