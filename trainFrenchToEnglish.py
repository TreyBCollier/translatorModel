import io
import os
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
import tensorflow as tf


from encoder import Encoder
from decoder import Decoder

filePath = "fra.txt"


def sentenceSubstitutions(sentence):
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except for letters and punctuation
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    return sentence


def sentencePreprocessing(sentence):
    unicodeCategory = 'Mn'
    strippedSentence = sentence.lower().strip()
    normalized = unicodedata.normalize('NFD', strippedSentence)
    sentence = ''.join(i for i in normalized
                       if unicodeCategory != unicodedata.category(i))

    # implementing spaces betwen punctuation
    sentence = sentenceSubstitutions(sentence)
    sentence = sentence.strip()

    # Adding tags to the sentence so it knows the start and end of each sentence
    sentence = '<s> ' + sentence + ' <e>'
    return sentence


def buildDataset(path, data):
    #  removing accents for the encoding
    pairs = [[sentencePreprocessing(sentence) for sentence in l.split(
        '\t')] for l in io.open(path, encoding='UTF-8').read().strip().split('\n')[:data]]
    # retuning pairs of words - [English , French]
    return zip(*pairs)

# derives tenor from the data - a generalisation of vectors


def getTensor(token, lang):
    tensor = token.texts_to_sequences(lang)
    # padding text
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor


def tokenizeData(lang):
    # tokenizing data
    tokenizedLanguage = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    tokenizedLanguage.fit_on_texts(lang)

    return tokenizedLanguage


# specifying the number of lines of the corpus to use
# for reasons relating to computing power, 100,000 was decided
num_examples = 100000
englishData, frenchData = buildDataset(filePath, num_examples)

loadedFrenchData = tokenizeData(frenchData)
loadedEnglishData = tokenizeData(englishData)
frenchTensor = getTensor(loadedFrenchData, frenchData)
englishTensor = getTensor(loadedEnglishData, englishData)


frenchData = loadedFrenchData

englishData = loadedEnglishData

# uses shape of the data tensor/vector to calculate the maximum length
englishDataLength = englishTensor.shape[1]
frenchDataLength = frenchTensor.shape[1]

# splitting training/testing data using the common 80/20 split
frenchTraining, frenchValue, englishTraining, englishValue = train_test_split(
    frenchTensor, englishTensor, test_size=0.2)

# get the slices of an array in the form of objects
dataset = tf.data.Dataset.from_tensor_slices(
    (frenchTensor, englishTensor)).shuffle(len(frenchTensor))
dataset = dataset.batch(64, drop_remainder=True)

# using the Encode and Decoder classes to  create new instances
encoder = Encoder(len(frenchData.word_index)+1, 256, 1024, 64)
decoder = Decoder(len(englishData.word_index)+1, 256, 1024, 64)


def calculateObjectLoss(real, pred):
    objectLoss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    overallLoss = objectLoss(real, pred)
    return overallLoss

# determines how far predicted values deviate from actual values


def calculateLoss(real, pred):
    overallLoss = calculateObjectLoss(real, pred)
    lossDataType = overallLoss.dtype
    overallLoss *= tf.cast(tf.math.logical_not(
        tf.math.equal(real, 0)), dtype=lossDataType)
    return tf.reduce_mean(overallLoss)


# implementing the Adam algorithm which is a stochastic gradient descent method
optimizer = tf.keras.optimizers.Adam()

# initialising a checkpoint to save the model at regualr intervals
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# Method executed at each step during training
@ tf.function
def epcohSteps(french, english, hiddenEncoding):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, hiddenEncoding = encoder(french, hiddenEncoding)
        hiddenDecoding = hiddenEncoding
        inputDecoding = tf.expand_dims(
            [englishData.word_index['<s>']] * 64, 1)

        # giving the taget to the decoder for the next input
        dataLength = english.shape[1]
        for i in range(1, dataLength):
            subData = english[:, i]
            # encoder output is passed to the decoder
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

# The following  is commented out after training for translating

# for epoch in range(10):
#     interval = 2
#     hiddenEncoding = encoder.hiddenStateInit()
#     overallLoss = 0

#     for (batch, (french, english)) in enumerate(dataset.take(len(frenchTraining)//64)):
#         batchLoss = epcohSteps(french, english, hiddenEncoding)
#         overallLoss = overallLoss + batchLoss

#     # model checkpoint saved every two epochs
#     if (epoch) % interval == 1:
#         checkpoint.save(file_prefix=os.path.join(
#             './training_checkpoints', "ckpt"))


def calculateInput(sentence):
    sentenceIn = [frenchData.word_index[i] for i in sentence.split(' ')]
    sentenceIn = tf.keras.preprocessing.sequence.pad_sequences([sentenceIn],
                                                               maxlen=frenchDataLength,
                                                               padding='post')
    sentenceIn = tf.convert_to_tensor(sentenceIn)

    return sentenceIn


def evaluate(sentence):
    result = ""
    # cleaning the sentence before evaluation
    sentence = sentencePreprocessing(sentence)
    formatSentence = calculateInput(sentence)
    hidden = [tf.zeros((1, 1024))]

    ouputEncoding, hiddenEncoding = encoder(formatSentence, hidden)

    hiddenDecoding = hiddenEncoding
    inputDecoding = tf.expand_dims([englishData.word_index['<s>']], 0)

    # at each setp, the previous prediction and hidden state are fed into the decoder
    for i in range(englishDataLength):
        predictions, hiddenDecoding, bahdanauWeights = decoder(inputDecoding,
                                                               hiddenDecoding,
                                                               ouputEncoding)

        prediction = tf.argmax(predictions[0]).numpy()
        result = result + englishData.index_word[prediction] + " "
        # we stop predicting when the end token is predicted
        if('<e>' == englishData.index_word[prediction]):
            return result

        # prediction is fed back into the model
        inputDecoding = tf.expand_dims([prediction], 0)

    return result


def translateFrenchToEnglish(sentence):
    # retrieving the most recent checkpoint of the saved model
    checkpoint.restore(tf.train.latest_checkpoint(
        './training_checkpoints'))
    result = evaluate(sentence)

    return result
