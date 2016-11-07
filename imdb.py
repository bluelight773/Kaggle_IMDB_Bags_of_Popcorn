# -*- coding: utf-8 -*-

"""Implementation of various sentiment classification approaches on the IMDB dataset.

Approaches implemented include relying on:
-Pre-trained GoogleNews Word2Vec Model for features + XGBoost Classifer
-New Word2Vec Model trained on the IMDB dataset for features + XGBoost Classifer
-Recurrent Neural Nets:
 Relied on various implementations including using Passage library, TensorFlow's SKFlow library, and Keras with
  Theano/TensorFlow backend
-Convolutional Neural Nets + Recurrent Neural Nets

Amongst these approaches, the best results were with an RNN using Passage library with an AUC of 0.966.
See below for more details on approaches and AUC scores.

This code should be run in Python 2.7 particularly to accommodate use of Passage library.
"""

# Imports for TF RNN
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import re
import logging
from functools import reduce
from operator import add

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup  # HTML to text
from nltk.corpus import stopwords  # String clearning
import nltk.data  # To load sentence tokenizer

# Word2Vec and XGBoost classifer
from gensim.models import Word2Vec
import xgboost

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import TensorFlowRNNClassifier

# Metrics
from sklearn.metrics import roc_auc_score

# For Passage-based RNN
# pip install passage
# Note that one might also need to make other changes to be able to use passage RNN, such as:
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3
from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import Tokenizer
import passage.utils

# joblib is used to store/load passage tokenizers and/or models
from sklearn.externals import joblib

import h5py

# Imports pertaining to using Keras neural net models

# Setting numpy random seed before any Keras' modules are imported to ensure consistent results
# This should at least be the case if Theano is used as the backend.
# Note that the default backend will likely be TensorFlow.
SEED = 1000
np.random.seed(SEED)

from keras.models import Sequential
from keras.layers import Dense as KDense, GRU, Dropout, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding as KEmbedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer as KTokenizer
from keras import backend
from keras.models import load_model


stop_words = set(stopwords.words('english'))

# Based on https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

HOME_DIR = "/home/bluelight/Kaggle/Kaggle_IMDB_Bags_of_Popcorn"
os.chdir(HOME_DIR)

# Set of supported models to build.  Simply set MODEL_TYPE to the desired model.
MODEL_TYPE = "Keras NN" ##
MODEL_TYPES = {"Google News W2V", "IMDB W2V", "TF RNN", "Passage RNN", "Passage Char RNN", "Keras NN"}
assert MODEL_TYPE in MODEL_TYPES, "Invalid Model Type"

# Below are AUC scores on test data for the various models, from best to worst:

# Passage RNN (word-level): 0.965632574938
# Note this may take hours to complete training.

# Keras NN relying on an Embeddings Layer + GRU Layer ?? + Dense Layer, with following p=0.2 dropouts:
# On input to Embedding layer, after Embedding layer, and after GRU layer
# TensorFlow backend - ?? though it should be noted that Keras did not, as of implementation time,
#                                     support ensuring consistent results on repeated runs.
# Theano backend - 0.949987991306
# See train_and_save_keras_tokenizer_and_nn_model for AUC numbers on other variations
# Note that training generally took 20 or fewer minutes

# IMDB W2V - Avg Word2Vec on words in reviews, using a new Word2Vec model built just based on the IMDB reviews
# Preprocessing applied to reviews was: text2html, removed punct/#s, lowercasing, removing stopwords
# 1) Current Implementation - W2V Model built on ALL (un/labeled train, test reviews) data: 0.871369770318
# 2) Variation - W2V Model built on ALL un/labeled train reviews data: 0.8683802376425263
# 3) Variation - W2V Model on ONLY train portion of labeledTrain data + unlabeled Train data reviews data:
#    0.8641419928415065

# Google News W2V - Avg Word2Vec on words in reviews, using pretrained GoogleNews Word2Vec Model
# 1) Current implementation - with preprocessing of reviews (text2html, removed punct/#s, removed stopwords):
#    0.8543855093070176
# 2) Variation - with some preprocessing of reviews (only removed stopwords): 0.847591196868
# 3) Variation - without preprocessing of reviews: 0.837822552631

# Current implementation - TF RNN - RNN using TensorFlow via SKFlow - min-freq: 40, 1000 steps, 1 layer, 50 embeddings,
# 100 max length.  Note that these values resemble those used in:
# github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_builtin_rnn_model.py
# except for increasing max length from 10 to 100 and steps from 100 to 1000, which seems to make sense for our
# purposes.
# Preprocessing: html2text, removing punct/#s, lowercasing:
# 0.829232696481
# Note that more layers (3), and embeddings (100) seemed to make negligible difference on AUC

IMDB_W2V_MODEL = "300features_40minwords_10context"
GN_W2V_MODEL = "GoogleNews-vectors-negative300.bin"

PASSAGE_RNN_MODEL = "passage_rnn_model.pkl"
PASSAGE_TOKENIZER = "passage_tokenizer.pkl"

PASSAGE_CHAR_RNN_MODEL = "passage_char_rnn_model.pkl"
PASSAGE_CHAR_TOKENIZER = "passage_char_tokenizer.pkl"

TF_RNN_MODEL_DIR = "tf_rnn_model"
TF_TOKENIZER = "tf_tokenizer.pkl"

KERAS_NN_MODEL = "keras_nn_model.h5"
KERAS_TOKENIZER = "keras_tokenizer.pkl"
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def html_to_text(html):
    """Return extracted text string from provided HTML string."""
    text = BeautifulSoup(html.strip(), "lxml")
    text = " ".join(text.strings)  # One may/not need to use strip()
    return text


def letters_only(text):
    """Return input string with only letters (no punctuation, no numbers)."""
    # It is probably worth experimenting with milder prepreocessing (eg just removing punctuation)
    return re.sub("[^a-zA-Z]", " ", text)


def text_to_wordlist(text, get_text_from_html=False, letters_only=True, lowercase=True, remove_stopwords=False):
    """Convert text to a sequence of words with the specified preprocessing applied.
    Returns a list of words.
    """

    # 1. Remove HTML
    # review_text = BeautifulSoup(review).get_text()
    if get_text_from_html:
        text = html_to_text(text)

    # 2. Remove non-letters
    if letters_only:
        text = letters_only(text)

    # 3. Convert words to lower case and split them
    if lowercase:
        text = text.lower()

    words = text.split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]

    # 5. Return a list of words
    return(words)


# -1 accounts for first row being run twice when map is used (by design)
reviews_converted_to_sentences = -1


def review_to_sentences(review_html, remove_stopwords=False):
    """Split a review into parsed sentences, applies preprocessing (html->text, removing punctuation/#s, lowercasing,
    and removing stopwords depending on argument), and returns a list of sentences, where each sentence is a list of
    words.
    """
    global sentence_tokenizer, reviews_converted_to_sentences

    # 1. Extract the text from the html using BeautifulSoup
    #    " ".join(parsed.strings) is used instead of parsed.get_text() as the former
    #    better ensures strings on two sides of a tag are separated rather than concatenated.
    review_text = html_to_text(review_html)

    # 2. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = sentence_tokenizer.tokenize(review_text)

    # 3. Loop over each sentence converting them to preprocessed (lowercase, no punctuation/#s) wordlists
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call text_to_wordlist to get a list of words - removing punctuation/#s, lowercasing is done too
            sentences.append(text_to_wordlist(raw_sentence,
                                              remove_stopwords=remove_stopwords))

    reviews_converted_to_sentences += 1
    if reviews_converted_to_sentences % 500 == 0:
        print("Sentence conversion done for {} reviews".format(
            reviews_converted_to_sentences))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def rnn_tokenizer_review_preprocess(review):
    """Preprocessing used before fitting/transforming RNN tokenizer - Html->text, remove punctuation/#s, lowercase."""
    return letters_only(html_to_text(review)).lower()


def train_and_save_passage_tokenizer_and_rnn_model(x_train, y_train, x_test, character_model=False):
    """Train and save Passage tokenizer and Passage RNN model.

    x_train and x_test should each be a series that's already been pre-preocessed: html->text, lowercase, removed
    punct/#s
    x_train+x_test are used to build the tokenizer.

    Note that character-based RNN is a work-in-progress and not actuallly implemented as of now.
    """

    # Note that we assume we have train/test reviews that had been preprocessed: html->text, lowercased, removed
    # punct/#s

    # Note in https://github.com/IndicoDataSolutions/Passage/blob/master/examples/sentiment.py they only
    # extract text from html, lowercase and strip (no punctuation removal)

    # Tokenization: Assign each word in the reviews an ID to be used in all reviews
    tokenizer = Tokenizer(min_df=10, max_features=100000, character=character_model)

    train_reviews_list = x_train.tolist()
    tokenizer.fit(train_reviews_list + x_test.tolist())

    # Tokenize training reviws (so can use to fit RNN model on)
    train_reviews_tokenized = tokenizer.transform(train_reviews_list)

    # Based on https://github.com/vinhkhuc/kaggle-sentiment-popcorn/blob/master/scripts/passage_nn.py which is based
    # on https://github.com/IndicoDataSolutions/Passage/blob/master/examples/sentiment.py

    # RNN Network:
    # -Each tokenized review will be converted into a sequence of words, where each word has an embedding representation
    # (256)
    # -RNN layer (GRU) attempts to find pattern in sequence of words
    # -Final dense layer is used as a logistic classifier to turn RNN output into a probability/prediction
    if not character_model:
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            # May replace with LstmRecurrent for LSTM layer
            GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                           init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]
    else:
        # Character-level RNN
        # Idea is to convert character tokenizations into one-hot encodings in which case
        # the embeddings layer is no longer needed
        train_reviews_tokenized = map(lambda r_indexes: pd.get_dummies(r_indexes,
                                                                       columns=range(tokenizer.n_features + 1)).values,
                                      train_reviews_tokenized)
        layers = [
            # May replace with LstmRecurrent for LSTM layer
            GatedRecurrent(size=100, activation='tanh', gate_activation='steeper_sigmoid',
                           init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]

    # RNN classifer uses Binary Cross-Entropy as the cost function
    classifier = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
    NUM_EPOCHS = 10
    # 10 epochs may take 10+ hours to run depending on machine
    classifier.fit(train_reviews_tokenized, y_train.tolist(), n_epochs=NUM_EPOCHS)

    # Store model and tokenizer
    if character_model:
        passage.utils.save(classifier, PASSAGE_CHAR_RNN_MODEL)
        _ = joblib.dump(tokenizer, PASSAGE_CHAR_TOKENIZER, compress=9)
    else:
        passage.utils.save(classifier, PASSAGE_RNN_MODEL)
        _ = joblib.dump(tokenizer, PASSAGE_TOKENIZER, compress=9)


def train_and_save_tf_tokenizer_and_rnn_model(x_train, y_train, x_test):
    """Train and return TensorFlow tokenizer (assigns ID per word so reviews can be represented as sequences of IDs)
    and TensorFlow RNN model.  Saving isn't currently supported due to issue with restoring SKFlow models:
    https://github.com/tensorflow/tensorflow/issues/3008

    x_train and x_test should each be a series that's already been pre-preocessed: html->text, lowercase, removed
    punct/#s.
    x_test is used to build the tokenizer.
    """
    global TF_RNN_EMBEDDING_SIZE, n_words, TF_RNN_MODEL_DIR
    # x_train/test will be a series of preprocessed reviews, y_train/test will be a series of 0/1
    # sentiment values
    x = pd.concat([x_train, x_test], axis=0, ignore_index=True)

    # Process vocabulary - whereby we convert X from a series of reviews to
    # a numpy array whose number of rows is the number of instances but whose
    # features (columns) correspond to the index of each word up to MAX_DOCUMENT_LENGTH
    # e.g. "this is a better plan to go than the last" becomes something like
    # [7, 2, 3, 234, 544, 75, 289, 123, 99, 149]

    # Based on:
    # github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_builtin_rnn_model.py
    MAX_DOCUMENT_LENGTH = 100  # Tried 10, 100 and 500. 100 was best.
    # Setting min_frequency seems to have helped ~4pct points (from 0 to 40)
    tokenizer = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=40)
    tokenizer.fit(x)

    x_train = np.array(list(tokenizer.transform(x_train)))  # list is used to convert an iterator

    n_words = len(tokenizer.vocabulary_)
    print('Total words: %d' % n_words)

    # 200 and above seem to make things worse
    # 100 may have produced better results when different preprocessing of text was done.  There is room to experiment
    # with different text preprocessing.
    TF_RNN_EMBEDDING_SIZE = 50

    # Build model: a bidrectional GRU with a single layer
    # One may try getting closer to the parameters used in the passage implementation to attempt to improve results

    # Setting num_steps to fit on from 100 to 1000 made a big difference  ~4pct more. 5000 may have caused a crash
    # 2000 steps got score down to 58

    # Depending on other variations in this implementation, increasing layers up to 3 may be helpful,
    # Incrasing number of steps from 100 to 1000 made a big difference (~4 pct points).  Going up to 2000 may
    # cause a decline, and 5000 steps caused a crash.
    NUM_LAYERS = 1
    NUM_STEPS = 1000

    # Setting bidirectional to True increased score by ~1 pct
    # Note that rnn_size could probably be set to other values and setting it to equal the embedding size is only
    # a suggestion.
    classifier = TensorFlowRNNClassifier(
        rnn_size=TF_RNN_EMBEDDING_SIZE, n_classes=2, cell_type='gru',
        input_op_fn=input_op_fn, num_layers=NUM_LAYERS, bidirectional=True,
        sequence_length=None, steps=NUM_STEPS, optimizer='Adam',
        learning_rate=0.01, continue_training=True)

    classifier.fit(x_train, y_train, steps=NUM_STEPS)

    # _ = joblib.dump(tokenizer, TF_TOKENIZER, compress=9)
    # classifier.save(TF_RNN_MODEL_DIR)
    return (tokenizer, classifier)


def train_and_save_keras_tokenizer_and_nn_model(x_train, y_train, x_test, use_cnn=True, dropout_variation=2): ##
    """Train and save Keras tokenizer and Keras NN model.  With default parameters, best results may be attained.

    Keyword arguments:
    x_train:            Should be a series that's already been pre-preocessed: html->text, lowercase, remove punct./#s
                        It is used:
                        1) Along with x_test to fit the tokenizer (converts reviews into sequences of integers
                           corresponding to word-frequency rankings).
                        2) Along with y_train to fit the classification neural net.
    y_train:               Should be a series of sentiment values.  It is used along with x_train to fit the
                           classification neural net.
    x_test:             Should be a series that's already been pre-preocessed: html->text, lowercase, remove punct./#s
                        It is also used along with x_train to fit the tokenizer (converts reviews into sequences of
                        integers corresponding to word-frequency rankings).
    use_cnn:            Used to indicate whether or not to include CNN layer (with the corresponding maxpool layer) in
                        the trained neural net.  Generally found to produce worse results during tests.  Default: False.
                        When set to False:
                        At 3 epochs, dropout_variation=0 and 20% of training data going to
                        validation, training acc was .9001 and validation acc was .8710.  With dropout_variation=2:
                        0.8596/0.8720 (Theano).

                        Setting to True:
                        At 3 epochs, dropout_variation=0 produced training acc of .9043 and validation
                        acc of .8550.  With dropout_variation=2, 0.8937/0.8792 (Theano).  Test AUC: 0.952308125317
                        in TensorFlow.
                        At 10 epochs, dropout_variation=0  produced training acc of .9843 and validation
                        acc of .8625. (overfitting).  With dropout_variation=2, ??/?? (Theano).  Test AUC 0.954923156362
                        in TensorFlow, 0.947220311445 in Theano.
    dropout_variation:  Indicates how to use dropout layers, if any, in the trained neural net as follows:
                        Unless specified otherwise, training and validation accuracy scores are reported in the form
                        training acc/validation acc, where number of epochs is 3, dropout_variation is 0, 20% of
                        training data is going to validation, and use_cnn is False.  Under those conditions, running
                        time on a modern local machine took 14 to 22 minutes.
                        0 = No dropouts. Generally found to produce the best results.
                            .9001/.8710.  When use_cnn is True:  .9043/.8550.
                            Test AUC: 0.946452427092 (TF) / 0.938564451483 (Theano)
                            When use_cnn is True at 10 epochs: .9843/.8625 (overfitting).
                            When use_cnn is False at 10 epochs: .9782/.8585 (overfitting)
                        1 = One p=0.5 dropout after GRU layer.  .9081/.8505.  Based on suggestions from:
                            http://www.icfhr2014.org/wp-content/uploads/2015/02/ICFHR2014-Bluche.pdf
                        2 = p=0.2 dropout between layers: on input to Embedding layer, after Embedding layer, and after
                            GRU layer.
                            No CNN: 0.8596/0.8720 (theano). With 10 epochs: 0.9301/0.8802
                                    Test AUC: 0.948833044597 (TF) / 0.949987991306 (Theano)
                            With CNN: 3 epochs: 0.8937/0.8792 (theano), Test AUC: 0.952308125317 (TF).
                                     10 epochs: ??/?? (Theano). Test AUC: 0.954923156362 (TF) / 0.947220311445 (Theano)
                            Based on suggestions from link below.
                        3 = Applies p=0.2 dropout to input to embeddings and p=0.2 dropout_W/U to input gates and
                            recurrent connections respectively in GRU layer.
                            0.8246/0.8465 (theano) With 10 epochs: 0.8929/0.8353
                            Based on suggestions from link below.
                        Default: 2.

    Approach largely based on:
    http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    """

    assert dropout_variation in (0, 1, 2, 3), "dropout_variation is not 0, 1, 2, or 3"

    np.random.seed(SEED)

    # Note that we assume we have train/test reviews preprocessed: html->text, lowercase, punct/#s removed
    # Note that in https://github.com/IndicoDataSolutions/Passage/blob/master/examples/sentiment.py they only
    # extract text from html, lowercase and strip (no punctuation/#s removal) in case one wants to experiment
    # with different pre-processing variations.

    # Tokenization: Assign each word in the reviews an ID corresponding to its frequency rank
    # Note only top 5000 most frequent words are included
    num_most_freq_words_to_include = 5000
    tokenizer = KTokenizer(nb_words=num_most_freq_words_to_include)

    # Need to convert unicode strings into ascii to avoid tokenization errors
    # Note that we use both training and test data to fit the tokenizer, since we're not making use of the
    # test target values, and could theoretically apply this approach at least if the sentiment prediction process
    # is done in batches offline.
    train_reviews_list = [s.encode('ascii') for s in x_train.tolist()]
    test_reviews_list = [s.encode('ascii') for s in x_test.tolist()]
    all_reviews_list = train_reviews_list + test_reviews_list

    tokenizer.fit_on_texts(all_reviews_list)

    # Tokenize reviews where the result is a [review1 tokenized into list of word-freq-ranks, review2 tokenized into..]
    train_reviews_tokenized = tokenizer.texts_to_sequences(train_reviews_list)
    # Commented out since we won't evaluate at the end of this function
    # test_reviews_tokenized = tokenizer.texts_to_sequences(test_reviews_list)

    # Truncate and pad input sequences, so that we only cover up to the first 500 tokens per review
    # This ensures all reviews have a representation of the same size, which is needed for the Keras NN to process them.
    x_train = sequence.pad_sequences(train_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

    # Commented out since we won't evaluate at the end of this function
    # x_test = sequence.pad_sequences(test_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

    # Create the neural net model, which roughly consists of the following:
    # Embedding layer:      Ensures each review is represented as a 32-entry vector whose values typically correspond to
    #                       semantic relationship with other words appearing in the reviews.
    # CNN + MaxPool layer:  Helps turn the representation corresponding to a sequence of words into a higher-level
    #                       representation corresponding to a sequence of multiple adjacent words.
    #                       Let's call this the conceptual sequence representation.
    # GRU (RNN) layer:      Helps turn the conceptual sequence representation into one corresponding to the sequential
    #                       relationship of elements in that representation.
    #                       Let's call this the conceptual sequence relationship representation.
    # Dense layer:          Fully connected layer, which with the help of the sigmoid function, can turn the
    #                       conceptual sequence relationship representation into a binary classification probability.
    # Depending on dropout_variation, dropout may be used in different parts of the neural net to help reduce
    # overfitting.

    # Indicate it's a sequential type of model - linear stack of layers
    model = Sequential()

    # Decide on dropout to apply (if any) to the input of the Embedding layer
    initial_dropout = 0.0  # Default KEmbedding dropout value (no dropout)
    if dropout_variation == 2 or dropout_variation == 3:
        initial_dropout = 0.2

    # Create a 32-entry word embedding - ie: each word will be mapped into being a 32-entry word embedding vector
    # Words beyond the most frequent num_most_freq_words_to_include (5000) or beyond the first
    # MAX_REVIEW_LENGTH_FOR_KERAS_RNN (500) in a review are discarded.
    embedding_vector_length = 32
    # Note we provide KEmbeddings with size of vocab (num_most_freq_words_to_include),
    # size of embedding vector (embedding_vector_length),
    # length of each input sequence (MAX_REVIEW_LENGTH_FOR_KERAS_RNN), and dropout to apply to input
    # Outputs a 3D Tensor of shape (# of samples, sequence/review length, embedding vector length)
    model.add(KEmbedding(num_most_freq_words_to_include, embedding_vector_length,
                         input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, dropout=initial_dropout))

    if dropout_variation == 2:
        model.add(Dropout(0.2))

    # Incorporate CNN and corresponding MaxPool layer
    if use_cnn:
        model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))  # Cuts representation size in half

    # Add GRU layer of size 100 units
    # Set dropout values for input units for input gates (dropout_W), for input units for recurrent connections
    # (dropout_U). Default values (when dropout_variation is 0) is 0.0.
    dropout_W = 0.0
    dropout_U = 0.0

    if dropout_variation == 3:
        dropout_W = 0.2
        dropout_U = 0.2

    model.add(GRU(100, dropout_W=dropout_W, dropout_U=dropout_U))

    # Add potential dropout:  This is based on recommendation for p=0.5 and placement after each GRU/LSTM layer:
    # http://www.icfhr2014.org/wp-content/uploads/2015/02/ICFHR2014-Bluche.pdf
    if dropout_variation == 1:
        model.add(Dropout(0.5))

    elif dropout_variation == 2:
        model.add(Dropout(0.2))

    # Add layer to get final probability prediction
    model.add(KDense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Run 3 epochs.  More is said to overfit - eg training acc increasing while validation acc stagnating or declining
    # after the 3rd epoch.
    # Note that while tuning, validation_split parameter was set to 0.2 to use last 20% of training data
    # to report validation score. It seems that if that parameter is set as such, only 80% of x/y_train would be used to
    # train.
    model.fit(x_train, y_train, nb_epoch=10, batch_size=64, validation_split=0.2) ##

    # Save model
    model.save(KERAS_NN_MODEL)

    # When Theano is used as the backend, an exception may occur when attempting to load a model that can be resolved by
    # deleting "optimizer_weights" in the model H5 file - see https://github.com/fchollet/keras/issues/4044
    if backend.backend() == "theano":
        with h5py.File(KERAS_NN_MODEL, "r+") as f:
            del f["optimizer_weights"]
    _ = joblib.dump(tokenizer, KERAS_TOKENIZER, compress=9)

    # Note following lines could be used to evaluate the model's accuracy against the test set
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))


def train_and_save_w2v_model():
    """Train a Word2Vec model based on the available IMDB reviews corpus (preprocessed - html->text, lowercased,
    remove punctuations/#s, stopwords kept) storing the model at the end.

    This implementation is based on https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
    """
    global IMDB_W2V_MODEL
    # Note ALL available reviews are used to build W2V model. We assume this is fine since
    # W2V does not use the labels, and since we're running things in an offline and batched manner.
    labeled_train_df = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, sep='\t')
    unlabeled_train_df = pd.read_csv('unlabeledTrainData.tsv', header=0, quoting=3, sep='\t')
    unlabeled_test_df = pd.read_csv('testData.tsv', header=0, quoting=3, sep='\t')

    # We'll get a series where each entry is a list of sentences - each sentence being a list of words
    labeled_train_sentences = labeled_train_df["review"].map(review_to_sentences)
    unlabeled_train_sentences = unlabeled_train_df["review"].map(review_to_sentences)
    unlabeled_test_sentences = unlabeled_test_df["review"].map(review_to_sentences)

    # This will be a series containing all entries from un/labeled data
    all_sentences = pd.concat([labeled_train_sentences, unlabeled_train_sentences, unlabeled_test_sentences],
                              axis=0, ignore_index=True)

    # Combine all the list of lists into one list of lists (ie: one list containing all sentences where
    # each sentence is a list of words)
    sentences = list(reduce(add, all_sentences.values))

    # Configure logging module so that Word2Vec creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters - see https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(IMDB_W2V_MODEL)


# -1 accounts for first row being run twice when map is used (by design)
reviews_done = -1

def get_w2v_features(review, get_text_from_html=True, letters_only=True, lowercase=True, remove_stopwords=True,
                     num_features=300):
    """Average all of the word vectors in the provided review after pre-processing it according to the parameter values.

    Based on https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
    """
    global stop_words, reviews_done, w2v_model, index2word_set

    # Note that one would expect appropriate and consistent preprocessing should be before feeding data for training w2v
    # model and before attempting to get w2v features.  However, empirical verification is the only way to know for sure
    # (eg when may be better off removing stopwords even if Google News model includes them)
    words = text_to_wordlist(review, get_text_from_html=get_text_from_html, letters_only=letters_only,
                             lowercase=lowercase,
                             remove_stopwords=remove_stopwords)

    # Pre-initialize an empty numpy array (for speed)
    # All-zeros will be what's returned if none of the words are in the w2v model
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Loop over each word in the review and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, w2v_model[word])

    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)

    reviews_done += 1
    if reviews_done % 500 == 0:
        print("Word2Vec features extracted for {} reviews".format(reviews_done))

    return feature_vec


def input_op_fn(x):
    """Customized function to transform batched x (input) into embeddings - list of tensors, each of which contains
    embeddings for all ith words in reviews (needed for TF RNN model)."""
    global n_words, TF_RNN_EMBEDDING_SIZE

    # x will be a tensor of size batch x sequence length (number of words in a review)
    # Convert x into embeddings.  word_vectors will thus be: batch_size x sequence_length x TF_RNN_EMBEDDING_SIZE
    word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
                                                  embedding_size=TF_RNN_EMBEDDING_SIZE, name='words')
    # word_list will be a list of size sequence-length (presumably max # of words in a review) where each element
    # will be a tensor of batch-size x embedding-size (eg: first element has embeddings for all the first words in
    # the reviews, the 2nd element has emebddings for the second words in the reviews, etc)
    word_list = tf.unpack(word_vectors, axis=1)
    return word_list


# Note that one would expect appropriate and consistent preprocessing should be before feeding data for training w2v
# model and before attempting to get w2v features.  However, empirical verification is the only way to know for sure
# (eg when may be better off removing stopwords even if Google News model includes them)

def get_w2v_features_via_gn_model(review):
    """Get GoogleNews Word2Vec features for the provided HTML non-preprocessed review."""
    # Note that this combination of pre-processing was found to produce the best results empirically
    # See https://groups.google.com/forum/#!topic/word2vec-toolkit/YMnVWxd0fTo for GoogleNews model preprocessing
    return get_w2v_features(review, get_text_from_html=True, letters_only=True, lowercase=False, remove_stopwords=True,
                            num_features=300)


def get_w2v_features_via_imdb_model(review):
    """Get IMDB Word2Vec features for the provided HTML non-preprocessed review."""
    # It may be better to set remove_stopwords to False since when IMDB model is built the stopwords aren't removed
    return get_w2v_features(review, get_text_from_html=True, letters_only=True, lowercase=True, remove_stopwords=True,
                            num_features=300)


def get_train_test_data(reviews_to_features_fn=None):
    """Extracts features (using reviews_to_features_fn), splits into train/test data, and returns
    x_train, y_train, x_test, y_test.  If no feature extraction function is provided, x_train/x_test will
    simply consist of a Series of all the reviews.
    """
    df = pd.read_csv('labeledTrainData.tsv', header=0, quotechar='"', sep='\t')

    # Shuffle data frame rows
    np.random.seed(SEED)
    df = df.iloc[np.random.permutation(len(df))]

    if reviews_to_features_fn:
        feature_rows = df["review"].map(reviews_to_features_fn)
        if type(feature_rows[0]) == np.ndarray:
            num_instances = len(feature_rows)
            num_features = len(feature_rows[0])
            x = np.concatenate(feature_rows.values).reshape((num_instances, num_features))
        else:
            x = feature_rows
    else:
        x = df["review"]

    y = df["sentiment"]

    # Split 80/20
    test_start_index = int(df.shape[0] * .8)
    x_train = x[0:test_start_index]
    y_train = y[0:test_start_index]
    x_test = x[test_start_index:]
    y_test = y[test_start_index:]

    return x_train, y_train, x_test, y_test

# Train and evaluate the selected model (MODEL_TYPE)
if __name__ == "__main__":
    global w2v_model, index2word_set
    if MODEL_TYPE == "Google News W2V":
        print("Loading GoogleNews Word2Vec Model")
        w2v_model = Word2Vec.load_word2vec_format(GN_W2V_MODEL, binary=True)
        get_features_from_select_w2v_model = get_w2v_features_via_gn_model
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed - used by get_w2v_features_via_imdb/gn_model
        index2word_set = set(w2v_model.index2word)
    elif MODEL_TYPE == "IMDB W2V":
        if not os.path.isfile(IMDB_W2V_MODEL):
            print("Building IMDB Word2Vec Model")
            train_and_save_w2v_model()
        else:
            print("Loading IMDB Word2Vec Model")
        w2v_model = Word2Vec.load(IMDB_W2V_MODEL)
        get_features_from_select_w2v_model = get_w2v_features_via_imdb_model
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed - used by get_w2v_features_via_imdb/gn_model
        index2word_set = set(w2v_model.index2word)

    if MODEL_TYPE == "TF RNN":
        x_train, y_train, x_test, y_test = get_train_test_data(rnn_tokenizer_review_preprocess)

        # Note that ideally, we perhaps could handle training/storing/loading model and tokenizer separately
        # Note: SKFlow seems to currently have a limitation where save/restore of RNN model doesn't seem to be
        # supported: https://github.com/tensorflow/tensorflow/issues/3008
        # and thus this code is commented out.
        # if not os.path.isdir(TF_RNN_MODEL_DIR) or not os.path.isfile(TF_TOKENIZER):
        #    print("Building TensorFlow Tokenizer and RNN Model")
        #    tokenizer, classifier = train_and_save_tf_tokenizer_and_rnn_model(x_train, y_train, x_test)
        # else:
        #    print("Loading TensorFlow RNN Model")

        print("Building TensorFlow Tokenizer and RNN Model")
        tokenizer, classifier = train_and_save_tf_tokenizer_and_rnn_model(x_train, y_train, x_test)

        # Commented out due to aforementioned issue with supporting save/restore of SKFlow-based models.
        #classifier = TensorFlowRNNClassifier.restore(TF_RNN_MODEL_DIR)
        #tokenizer = joblib.load(TF_TOKENIZER)

        # Before passing the reviews to the classifier for prediction, they need to be tokenized (into list of IDs)
        # using the tokenizer created during training
        test_reviews_tokenized = np.array(list(tokenizer.transform(x_test)))
        y_predict = classifier.predict(test_reviews_tokenized)

    # RNN using Passage library - best results
    elif MODEL_TYPE == "Passage RNN" or MODEL_TYPE == "Passage Char RNN":
        if MODEL_TYPE == "Passage RNN":
            passage_rnn_model = PASSAGE_RNN_MODEL
            passage_tokenizer = PASSAGE_TOKENIZER
        elif MODEL_TYPE == "Passage Char RNN":
            passage_rnn_model = PASSAGE_CHAR_RNN_MODEL
            passage_tokenizer = PASSAGE_CHAR_TOKENIZER

        x_train, y_train, x_test, y_test = get_train_test_data(rnn_tokenizer_review_preprocess)

        # Note that ideally, we perhaps could handle training/storing/loading model and tokenizer separately
        if not os.path.isfile(passage_rnn_model) or not os.path.isfile(passage_tokenizer):
            print("Building Passage Tokenizer and RNN Model")
            train_and_save_passage_tokenizer_and_rnn_model(x_train, y_train, x_test,
                                                           character_model=MODEL_TYPE == "Passage Char RNN")
        else:
            print("Loading Passage RNN Model")

        classifier = passage.utils.load(passage_rnn_model)
        tokenizer = joblib.load(passage_tokenizer)

        # Before passing the reviews to the classifier for prediction, they need to be tokenized (into list of IDs)
        # using the tokenizer created during training
        test_reviews_tokenized = tokenizer.transform(x_test)

        print("Making predictions...")
        y_predict = classifier.predict(test_reviews_tokenized)

    # Keras NN, which will by default rely on an Embeddings Layer + GRU Layer + Dense Layer only
    elif MODEL_TYPE == "Keras NN":
        # Get train/test split with reviews having been preprocessed (html->text, remove punctuation/#s, lowercase)
        x_train, y_train, x_test, y_test = get_train_test_data(rnn_tokenizer_review_preprocess)

        # Note that ideally, we perhaps could handle training/storing/loading model and tokenizer separately
        if not os.path.isfile(KERAS_NN_MODEL) or not os.path.isfile(KERAS_TOKENIZER):
            print("Building Keras Tokenizer and NN Model")
            train_and_save_keras_tokenizer_and_nn_model(x_train, y_train, x_test)
        else:
            print("Loading Keras NN Model")

        classifier = load_model(KERAS_NN_MODEL)
        tokenizer = joblib.load(KERAS_TOKENIZER)

        # Prepare test data for prediction - convert reviews to ascii, tokenize and pad reviews
        test_reviews_list = [s.encode('ascii') for s in x_test.tolist()]
        test_reviews_tokenized = tokenizer.texts_to_sequences(test_reviews_list)
        test_reviews_tokenized_padded = sequence.pad_sequences(test_reviews_tokenized,
                                                               maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

        print("Making predictions...")
        y_predict = classifier.predict(test_reviews_tokenized_padded)

    # "Google News W2V" and "IMDB W2V" rely on an XGBoost classifier trained on their features
    else:
        print("Preparing Training/Test Data for Classifier")
        x_train, y_train, x_test, y_test = get_train_test_data(get_features_from_select_w2v_model)
        print("Training classifier...")
        classifier = xgboost.XGBClassifier(seed=SEED,
                                           learning_rate=0.01, n_estimators=1000, subsample=0.5,
                                           max_depth=6)
        classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_test)

    # Note that given we're computing the threshold-based ROC score, we should have y_predict be probabilities
    print("AUC: {}".format(roc_auc_score(y_test, y_predict, average='macro')))


