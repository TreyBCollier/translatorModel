

import tensorflow as tf

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

from encoder import Encoder
from decoder import Decoder

path_to_file = "fra.txt"


def sentencePreprocessing(sentence):
    sentence = ''.join(c for c in unicodedata.normalize('NFD', sentence.lower().strip())
                       if unicodedata.category(c) != 'Mn')

    # implementing spaces betwen punctuation
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    sentence = sentence.strip()

    # Adding tags to the sentence so it knows the start and end of each sentence
    sentence = '<s> ' + sentence + ' <e>'
    return sentence


def buildDataset(path, data):
    # sample = io.open(path, encoding='UTF-8').read().strip().split('\n')
    pairs = [[sentencePreprocessing(sentence) for sentence in l.split(
        '\t')] for l in io.open(path, encoding='UTF-8').read().strip().split('\n')[:data]]
    return zip(*pairs)


def getTensor(token, lang):
    tensor = token.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor


def tokenizeData(lang):
    tokenizedLanguage = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizedLanguage.fit_on_texts(lang)

    tensor = getTensor(tokenizedLanguage, lang)

    return tensor, tokenizedLanguage


def getData(file, data=None):
    # creating cleaned input, output pairs
    englishData, frenchData = buildDataset(file, data)

    frenchTensor, frenchTokenizer = tokenizeData(frenchData)
    englishTensor, englishTokenizer = tokenizeData(englishData)

    return [frenchTensor, englishTensor, frenchTokenizer, englishTokenizer]


# Try experimenting with the size of that dataset
num_examples = 100000
loadedData = getData(
    path_to_file, num_examples)

frenchTensor = loadedData[0]
englishTensor = loadedData[1]
frenchData = loadedData[2]
englishData = loadedData[3]

# Calculate max_length of the target tensors
englishDataLength = englishTensor.shape[1]
frenchDataLength = frenchTensor.shape[1]

# Creating training and validation sets using an 80-20 split
frenchTraining, frenchValue, englishTraining, englishValue = train_test_split(
    frenchTensor, englishTensor, test_size=0.2)


dataset = tf.data.Dataset.from_tensor_slices(
    (englishTensor, frenchTensor)).shuffle(len(englishTensor))
dataset = dataset.batch(64, drop_remainder=True)

encoder = Encoder(len(englishData.word_index)+1, 256, 1024, 64)
decoder = Decoder(len(frenchData.word_index)+1, 256, 1024, 64)


optimizer = tf.keras.optimizers.Adam()


def calculateObjectLoss(real, pred):
    objectLoss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    overallLoss = objectLoss(real, pred)
    return overallLoss


def calculateLoss(real, pred):
    overallLoss = calculateObjectLoss(real, pred)
    lossDataType = overallLoss.dtype
    overallLoss *= tf.cast(tf.math.logical_not(
        tf.math.equal(real, 0)), dtype=lossDataType)
    return tf.reduce_mean(overallLoss)


checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(english, french, hiddenEncoding):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, hiddenEncoding = encoder(english, hiddenEncoding)

        hiddenDecoding = hiddenEncoding

        inputDecoding = tf.expand_dims(
            [frenchData.word_index['<s>']] * 64, 1)

        # Teacher forcing - feeding the target as the next input
        dataLength = french.shape[1]

        for i in range(1, dataLength):
            subData = french[:, i]
            # passing enc_output to the decoder
            predictions, hiddenDecoding, _ = decoder(
                inputDecoding, hiddenDecoding, enc_output)

            loss = loss + calculateLoss(subData, predictions)

            # using teacher forcing
            inputDecoding = tf.expand_dims(subData, 1)

    batchLoss = (loss / int(dataLength))

    decoderVars = decoder.trainable_variables
    encoderVars = encoder.trainable_variables

    variables = (decoderVars + encoderVars)

    optimizer.apply_gradients(zip(tape.gradient(loss, (variables)), variables))

    return batchLoss


# for epoch in range(10):

#     hiddenEncoding = encoder.hiddenStateInit()
#     total_loss = 0

#     for (batch, (english, french)) in enumerate(dataset.take(len(englishTraining)//64)):
#         batchLoss = train_step(english, french, hiddenEncoding)
#         total_loss = total_loss + batchLoss

#     # saving (checkpoint) the model every 2 epochs
#     if (epoch + 1) % 2 == 0:
#         checkpoint.save(file_prefix=os.path.join(
#             './training_checkpoints', "ckpt"))


def calculateInput(sentence):
    sentenceIn = [englishData.word_index[i] for i in sentence.split(' ')]
    sentenceIn = tf.keras.preprocessing.sequence.pad_sequences([sentenceIn],
                                                               maxlen=englishDataLength,
                                                               padding='post')
    sentenceIn = tf.convert_to_tensor(sentenceIn)

    return sentenceIn


def evaluate(sentence):
    result = ""
    sentence = sentencePreprocessing(sentence)

    formatSentence = calculateInput(sentence)

    hidden = [tf.zeros((1, 1024))]
    ouputEncoding, hiddenEncoding = encoder(formatSentence, hidden)

    hiddenDecoding = hiddenEncoding
    inputDecoding = tf.expand_dims([frenchData.word_index['<s>']], 0)

    for i in range(frenchDataLength):
        predictions, hiddenDecoding, bahdanauWeights = decoder(inputDecoding,
                                                               hiddenDecoding,
                                                               ouputEncoding)

        # storing the attention weights to plot later on

        prediction = tf.argmax(predictions[0]).numpy()

        result = result + frenchData.index_word[prediction] + " "

        if('<e>' == frenchData.index_word[prediction]):
            return result

        # the predicted ID is fed back into the model
        inputDecoding = tf.expand_dims([prediction], 0)

    return result


def translateEnglishToFrench(sentence):
    checkpoint.restore(tf.train.latest_checkpoint(
        './training_checkpointsEnglishToFrench'))
    result = evaluate(sentence)

    return result
