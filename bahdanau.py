import tensorflow as tf


class Bahdanau(tf.keras.layers.Layer):
    unitValue = 1024

    def __init__(self, unitValue):
        super(Bahdanau, self).__init__()
        # Defining the densely-connected neural-network layers
        self.V = tf.keras.layers.Dense(1)
        self.W1 = tf.keras.layers.Dense(1024)
        self.W2 = tf.keras.layers.Dense(1024)

    def call(self, bahdanaQuery, bahdanavValues):
        hidenStateShape = tf.expand_dims(bahdanaQuery, 1)
        score = self.V(tf.nn.tanh(
            self.W2(bahdanavValues) + self.W1(hidenStateShape)))
        # softmax typically applies to the last axis
        # we want to assign a weight to each input so we apply softmax on the first axis
        bahdanauWeights = tf.nn.softmax(score, axis=1)
        # vector passed into the GRU during decoding
        bahdanauVector = tf.reduce_sum(
            bahdanauWeights * bahdanavValues, axis=1)

        return bahdanauVector, bahdanauWeights

# Inspired and modified from TensorFlow example
# TensorFlow Addons Networks : Sequence-to-Sequence NMT with Attention Mechanism
# 2021
