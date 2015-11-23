#!/usr/bin/env python
# -*- coding: utf-8 -*-
from basics import loader as mnist
import code
import math
import numpy as np
import theano
from theano import tensor as T

__author__ = 'krekle'

##############################
##                          ##
##       Theano Config      ##
##                          ##
##############################
# Environment flags:
# -> "floatX=float32,device=gpu0,nvcc.fastmath=True"

# After Init Config
theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'
theano.config.mode = 'FAST_RUN'  # FAST_COMPILE

##############################
##                          ##
##       Numpy Config       ##
##                          ##
##############################

np.set_printoptions(threshold=np.nan)


class ANN():
    def score(self, percent, k=4):
        total = math.ceil(5 - ((100 - percent) / 4))
        return total

    ##############################
    ##                          ##
    ##   Activation Functions   ##
    ##                          ##
    ##############################

    def sigmoid(self, layer, weight):
        return T.nnet.sigmoid(T.dot(layer, weight))

    def rectify(self, layer, weight):
        # Tann.relu
        return T.maximum(T.dot(layer, weight), 0.)

    def soft(self, layer, weight):
        return T.nnet.softmax(T.dot(layer, weight))

    ##############################
    ##                          ##
    ##    Weight Improvement    ##
    ##                          ##
    ##############################

    # Stochastic Gradient Decent
    def sgd(self, cost, params, lr=0.05):
        gradients = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, gradients):
            updates.append([p, p - g * lr])
        return updates

    # RMS prop
    def rmsprop(self, error, params, rho=0.9, learning_rate=0.01):
        grads = T.grad(cost=error, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + 0.000001)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - learning_rate * g))
        return updates

    ##############################
    ##                          ##
    ##          Error           ##
    ##                          ##
    ##############################

    def cross_entropy(self, model, known):
        return T.mean(T.nnet.categorical_crossentropy(model, known))

    def squared_sum(self, model, known):
        return T.mean(T.sqr(model - known))

    ##############################
    ##                          ##
    ##      Equation Model      ##
    ##                          ##
    ##############################

    # Black box of computation
    # This is where the activation functions are applied
    # Represents what happens in a neuron
    def dynamic_model(self, start, node_width):
        """
        Function representing the whole network with neurons and synapses.
        Dynamically creates start to hidden, (hidden to hidden)*x and hidden to output.
        :param start: Start state
        :param node_width: Node width
        :return: a function model of the whole network
        """
        # Dynamic amount of hidden layers, skip output
        hiddens = []
        for i in range(0, len(node_width) - 1):
            if i == 0:
                # Input -> Hidden
                hiddens.append(self.rectify(start, node_width[i]))
                # hiddens.append(self.sigmoid(start, node_width[i]))
            else:
                # Hidden -> Hidden
                hiddens.append(self.rectify(hiddens[i - 1], node_width[i]))

        # Hidden to output -> Cost function (softmax) Last Layer
        hiddens.append(self.soft(hiddens[-1], node_width[-1]))
        return hiddens

    ##############################
    ##                          ##
    ##         Training         ##
    ##                          ##
    ##############################

    def train(self, epochs=20, batch=128, verbose_level=2):
        """
        Method for training the neural network
        :param epochs: Number of runs to loop over the training data
        :param batch: How many Xs to run at the same time
        :param verbose_level: Verbose Level, higher = more output
        :return:
        """
        for i in range(epochs):
            error = 0

            if verbose_level >= 1:
                print('Training ... [Epoch {level} / {total}]'.format(level=i + 1, total=epochs))

            for start, end in zip(range(0, len(self.trainX), batch),
                                  range(batch, len(self.trainX), batch)):
                error += self.training(self.trainX[start:end], self.trainY[start:end])

            # Prediction
            if verbose_level >= 1:
                print('- error Rate: {error}'.format(error=error))

            #solutions = np.argmax(self.testY, axis=1)
            guessed = self.predicting(self.testX)

            #correct = 0
            #for i in range(len(guessed)):
            #    if guessed[i] == solutions[i]:
            #        correct += 1

            #if verbose_level >= 2:
            #    print('- guessed {correct} of {total} ({percent}%)'.format(correct=correct, total=len(solutions),
            #                                                               percent=(correct / (len(solutions) / 100))))

    ##############################
    ##                          ##
    ##         Predict          ##
    ##                          ##
    ##############################

    def blind_test(self, question, toList=True):
        """
        Method for predicting one or multiple inputs to outputs
        :param question: Item(s) to predict
        :param toList: Return values as numpy.array or list
        :return: A list or numpy.array of the predicted values
        """
        predictor = None
        # Type convertion if question is single item
        if type(question) is list or type(question) is np.ndarray:
            predictor = question
        else:
            predictor = [question]

        # Actual prediction as numpy.array
        prediction = self.predicting(predictor)

        # Returns as numpy.array or list
        if toList:
            return prediction.tolist()
        else:
            return prediction

    ##############################
    ##                          ##
    ##      Initialize ANN      ##
    ##                          ##
    ##############################

    def __init__(self, data=None, nodes=[784, 625, 10]):
        """
        Initializes an Neural Network

        :param data:  The training and test data, can also be set with set_sets()
        :param nodes: The layers, first must be number of input features ex. mnist 28x28
        is 784. After comes hidden layers. Then output 10 nodes -> 0-9 mnist labels
        :return:
        """

        if data:
            self.trainX, self.trainY, self.testX, self.testY = data
        else:
            self.trainX, self.testX, self.trainY, self.testY = mnist.mnist(indexed=True)

        ## Theano Variables ##
        self.unknown = T.fmatrix()
        self.known = T.fmatrix()

        ### Synapse Weights ###
        self.layers = []
        # Synapses and Neurons
        if len(nodes) < 2:
            raise ValueError('Some nodes are required')
        else:
            for i in range(1, len(nodes)):
                self.layers.append((theano.shared(np.random.randn(nodes[i - 1], nodes[i]) * 0.01)))

        # Network model -> Activation Functions
        self.equation_model = self.dynamic_model(self.unknown, self.layers)

        # index of highest in output, ie: the digit with highest probability
        #self.predicted_index = T.argmax(self.equation_model, axis=1)
        self.predicted_index = T.argsort(self.equation_model[-1], axis=1)


        # Error Function, crossentropy of predicted and actual
        self.error = self.cross_entropy(self.equation_model[-1], self.known)

        # Weight Improvement Function
        self.updates = self.rmsprop(self.error, self.layers)

        # Train function, output through error function, update with update
        self.training = theano.function(inputs=[self.unknown, self.known], outputs=self.error, updates=self.updates,
                                        allow_input_downcast=True)

        # Predict function
        #self.predicting = theano.function(inputs=[self.unknown], outputs=self.predicted_index,
        #                                  allow_input_downcast=True)
        self.predicting = theano.function(inputs=[self.unknown], outputs=self.predicted_index, allow_input_downcast=True)

    def get_tests(self):
        """
        Method for getting tests
        :return: The test Xs and Ys
        """
        return self.testX, self.testY

    def get_trains(self):
        """
        Method for getting training sets
        :return: The training Xs and Ys
        """
        return self.trainX, self.trainY

    def set_sets(self, train=None, test=None):
        """
        Setter for training and test sets. Set it before calling training and train the network
        :param train: The training X and Ys packaged together eg. [train_x, train_y]
        :param test: The test Xs and Ys packaged together eg. [test_x, test_y]
        :return:
        """
        if train:
            self.trainX, self.trainY = train
            print('Train updated')
        if test:
            self.testX, self.testY = test
            print('Test updated')

    def load_flat(self, filename='demo_prep'):
        return mnist.load_cases(filename)

