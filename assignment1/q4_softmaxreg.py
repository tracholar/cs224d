import numpy as np
import random

from cs224d.data_utils import *

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params

def getSentenceFeature(tokens, wordVectors, sentence):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # - tokens: a dictionary that maps words to their indices in
    #          the word vector list
    # - wordVectors: word vectors (each row) for all tokens
    # - sentence: a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    idxs = [tokens[w] for w in sentence]
    sentVector = np.mean(wordVectors[idxs, :], axis=0)
    ### END YOUR CODE

    return sentVector

def softmax_cost_func():
    import theano
    from theano import tensor as T
    W = T.dmatrix('W')
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    N = T.scalar('N')
    r = T.scalar('r')
    yp = T.nnet.softmax(T.dot(x, W))
    J = - T.sum(T.log(yp) * y) / N + 0.5 * r * T.sum(W * W)
    dJ = T.grad(J, [W])
    fJ = theano.function([W, x, y, N, r], J)
    fdJ = theano.function([W, x, y, N, r], dJ)
    fP = theano.function([W, x], T.argmax(yp, 1))
    return fJ, fdJ, fP

fJ, fdJ, fP = softmax_cost_func()

def softmaxRegression(features, labels, weights, regularization = 0.0, nopredictions = False):
    """ Softmax Regression """
    # Implement softmax regression with weight regularization.

    # Inputs:
    # - features: feature vectors, each row is a feature vector
    # - labels: labels corresponding to the feature vectors
    # - weights: weights of the regressor
    # - regularization: L2 regularization constant

    # Output:
    # - cost: cost of the regressor
    # - grad: gradient of the regressor cost with respect to its
    #        weights
    # - pred: label predictions of the regressor (you might find
    #        np.argmax helpful)

    # prob = softmax(features.dot(weights))
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    # cost = np.sum(-np.log(prob[range(N), labels])) / N
    # cost += 0.5 * regularization * np.sum(weights ** 2)

    ### YOUR CODE HERE: compute the gradients and predictions
    ground_truth = np.zeros((features.shape[0], weights.shape[1]))
    ground_truth[range(N), labels] = 1
    cost = fJ(weights, features, ground_truth, N, regularization)
    grad, = fdJ(weights, features, ground_truth, N, regularization)
    pred = fP(weights, features)
    ### END YOUR CODE

    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization = 0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights,
        regularization)
    return cost, grad

def sanity_check():
    """
    Run python q4_softmaxreg.py.
    """
    random.seed(314159)
    np.random.seed(265)

    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
    dimVectors = wordVectors.shape[1]

    dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
    dummy_features = np.zeros((10, dimVectors))
    dummy_labels = np.zeros((10,), dtype=np.int32)
    for i in xrange(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
    print "==== Gradient check for softmax regression ===="
    gradcheck_naive(lambda weights: softmaxRegression(dummy_features,
        dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

    print "\n=== Results ==="
    print softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0)

if __name__ == "__main__":
    sanity_check()
