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

    def score(self, percent):
        total = math.ceil(5 - ((100-percent) / 4))
        print(total)
        return total


    ##############################
    ##                          ##
    ##   Activation Functions   ##
    ##                          ##
    ##############################

    def sigmoid(self, x, w):
        return T.nnet.sigmoid(T.dot(x, w))

    def rectify(self, x):
        # Tann.relu
        return T.maximum(x, 0.)

    def soft(self, h, w):
        return T.nnet.softmax(T.dot(h, w))

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
    def rmsprop(self, error, params, rho=0.9, learning_rate=0.001):
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
                hiddens.append(self.rectify(T.dot(start, node_width[i])))
                # hiddens.append(self.sigmoid(start, node_width[i]))
            else:
                # Hidden -> Hidden
                hiddens.append(self.rectify(T.dot(hiddens[i - 1], node_width[i])))

        # Hidden to output -> Cost function (softmax) Last Layer
        pyx = self.soft(hiddens[-1], node_width[-1])
        return pyx

    ##############################
    ##                          ##
    ##         Training         ##
    ##                          ##
    ##############################

    def training(self, epochs=20, batch=128, verbose_level=1):
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

                error += self.train(self.trainX[start:end], self.trainY[start:end])

            # Prediction
            if verbose_level >= 1:
                print(error)

            solutions = np.argmax(self.testY, axis=1)
            guessed = self.predict(self.testX)

            correct = 0
            for i in range(len(guessed)):
                if guessed[i] == solutions[i]:
                    correct += 1

            if verbose_level >=2:
                print('Score: {score}'.format(score=self.score(correct / (len(solutions)/100))))
            print('Guessed {correct} of {total}'.format(correct=correct, total=len(solutions)))

            # print(i, str(np.mean( == )*100)+'%')

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
        prediction = self.predict(predictor)

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

        self.node_width = []
        # Synapses and Neurons
        if len(nodes) < 2:
            raise ValueError('Some nodes are required')
        else:
            for i in range(1, len(nodes)):
                print('Node ({x},{y})'.format(x=nodes[i - 1], y=nodes[i]))
                self.node_width.append((theano.shared(np.random.randn(nodes[i - 1], nodes[i]) * 0.01)))

        ## Theano Functions ##

        # Network model -> Activation Functions
        self.equation_model = self.dynamic_model(self.unknown, self.node_width)

        # index of highest in output, ie: the digit with highest probability
        self.predicted_index = T.argmax(self.equation_model, axis=1)

        # Error Function, crossentropy of predicted and actual
        self.error = self.cross_entropy(self.equation_model, self.known)

        # Weight Improvement Function
        self.updates = self.rmsprop(self.error, self.node_width)

        # Train function, output through error function, update with update
        self.train = theano.function(inputs=[self.unknown, self.known], outputs=self.error, updates=self.updates,
                                     allow_input_downcast=True)

        # Predict function
        self.predict = theano.function(inputs=[self.unknown], outputs=self.predicted_index, allow_input_downcast=True)

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
            self.testX, self.testY = testann
            print('Test updated')

    def load_flat(self, filename='demo_prep'):
        return mnist.load_cases(filename)


if __name__ == "__main__":

    ## LOAD gamelogic data

    # Load training data
    data_2048 = np.loadtxt('log-2048.txt', dtype=int, usecols=range(17))

    # Get the labels
    raw_labels_2048 = data_2048[:,16]
    labels_2048 = [[float(0) for y in range(4)] for x in range(len(raw_labels_2048))]
    for i in range(len(raw_labels_2048)):
        labels_2048[i][int(raw_labels_2048[i])] = 1.0
    labels_2048 = np.array(labels_2048)
    print(labels_2048)

    # Get the states
    states_2048 = np.delete(data_2048,np.s_[-1:],1)

    data = [Processstates_2048, labels_2048, states_2048, labels_2048]

    print('############################################################')
    print('##                 Starting Neural Network                ##')
    print('############################################################')
    print('#')
    ann = ANN(data=data, nodes=[16, 700, 32, 4])
    print('# Network started with layers: 16, 12, 12, 4 You now have \n'
          '# control of the neural network object ref: ann')
    print('#')
    print('############################################################')
    print('#')
    print('# To start with different layers declare a new: \n'
          '# ANN(nodes=[x, y, z]) or pass command line arguments')
    print('#')
    print('############################################################')
    print('#')
    print('# To start training type: \n'
          '# ann.training(epochs=[20], batch=[128], verbose_level=[1])')

    print('#')
    print('############################################################')
    print('#')
    print('# When You are satisfied with training, you can blind_test with \n'
          '# ann.blind_test(feature_set) Returns predictions in a list')
    print('#')
    print('############################################################')
    print('')

    # Start interactive shell
    code.interact(local=locals())
