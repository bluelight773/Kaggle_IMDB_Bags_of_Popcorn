# Kaggle_IMDB_Bags_of_Popcorn
An implementation of various models for sentiment classification (positive vs negative) of IMDB reviews based on the dataset used by the Kaggle Bags of Popcorn Knowledge Competition: https://www.kaggle.com/c/word2vec-nlp-tutorial.

Implemented Approaches include the following, from best to worst, with AUC score shown based on a random 20% Test Dataset.  See the code for parameter details as well as info on other variations of these models:
1. Bidirectional Recurrent Neural Net using Passage Library: 0.965632574938
2. Convolutional Neural Net (CNN) + RNN with Dropout using Keras: 0.953867095363 Theano Backend, 0.952308125317 TensorFlow Backend
3. Average Word2Vec Features Model based on IMDB Dataset: 0.871369770318
4. Average Word2Vec Features Models based on Google News Dataset (Pre-trained): 0.8543855093070176
5. Recurrent Neural Net (RNN) using TensorFlow's not fully matured at the time SKFlow: 0.829232696481
