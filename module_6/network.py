__author__ = 'krekle'

import theano
from theano import tensor as T
import numpy as np
from basics import mnist_basics as mnist

##############################
##                          ##
##       Theano Config      ##
##                          ##
##############################

theano.config.optimizer = 'None' #'fast_compile'
theano.config.exception_verbosity = 'high'


# theano.config.compute_test_value = 'warn'


class ANN():
    def gen_weights(self, shape):
        return theano.shared(np.random.randn(*shape) * 0.01)

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

    def rmsprop(self, error, params, rho=0.9, learning_rate=0.001):
        # TODO : Dynamic learning rate?

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

        # Dynamic amount of hidden layers, skip output
        hiddens = []
        for i in range(0, len(node_width) - 1):
            if i == 0:
                # Input -> Hidden
                hiddens.append(self.rectify(T.dot(start, node_width[i])))
                #hiddens.append(self.sigmoid(start, node_width[i]))
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

    def training(self, epochs=20, batch=16):
        for i in range(epochs):
            error = 0
            print('Training ... [Epoch {level}]'.format(level=i + 1))
            for start, end in zip(range(0, len(self.trainX), batch),
                                  range(batch, len(self.trainX), batch)):
                # print('Training ... [Image {image}]'.format(image=start))
                error += self.train(self.trainX[start:end], self.trainY[start:end])

            # Prediction
            print(error)
            solutions = np.argmax(self.testY, axis=1)
            guessed = self.predict(self.testX)
            correct = 0
            for i in range(len(guessed)):
                if guessed[i] == solutions[i]:
                    correct += 1

            print(correct)

            # print(i, str(np.mean( == )*100)+'%')

    ##############################
    ##                          ##
    ##         Predict          ##
    ##                          ##
    ##############################

    def blind_test(self, question):
        prediction = self.predict(question)
        print(prediction)
        return prediction

    ##############################
    ##                          ##
    ##      Initialize ANN      ##
    ##                          ##
    ##############################

    def __init__(self, data=None, nodes=[16, 16, 32, 64, 32, 16, 8, 4]):
        # Something with data
        if data:
            self.trainX, self.trainY, self.testX, self.testY = data
            print('got data')
        else:
            self.trainX, self.testX, self.trainY, self.testY = mnist.mnist(onehot=True)

        print('#### X')
        print(self.trainX)
        print('#### Y')
        print(self.trainY)

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
                self.node_width.append(self.gen_weights((nodes[i - 1], nodes[i])))

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
        return self.testX, self.testY

    def set_sets(self, train, test, trX=None, trY=None):
        if train:
            self.trainX, self.trainY = np.hsplit(train, 1)
            print('Train updated')
        if test:
            self.testX, self.trainY = np.hsplit(test, 1)
            print('Test updated')

        if trX and trY:
            self.trainX = trX
            self.trainY = trY


## LOAD 2048 data

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

data = [states_2048, labels_2048, states_2048, labels_2048]

print('Starting Neural Network')
n = ANN(data=data)


print('##### Training #####')
n.training()
