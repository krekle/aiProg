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

theano.config.optimizer = 'fast_compile'
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
    ##      Cost Functions      ##
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
    ##          Model           ##
    ##                          ##
    ##############################

    # Black box of computation
    # This is where the activation functions are applied
    # Represents what happens in a neuron
    def dynamic_model(self, inWeight, weights):

        # Output weight and node
       # w_o = weights.pop(-1)

        # Dynamic amount of hidden layers
        hiddens = []
        for i in range(0, len(weights) -1):
            if i == 0:
                hiddens.append(self.rectify(T.dot(inWeight, weights[i])))
            else:
                hiddens.append(self.rectify(T.dot(hiddens[i - 1], weights[i])))

        # Hidden to output -> Cost function (softmax) Last Layer
        pyx = self.soft(hiddens[-1], weights[-1])
        return pyx

    def no_dynamic_model(self, X, w_h, w_h2, w_o):
        print(w_h)
        # Input to hidden -> Activation method (sigmoid) Layer 1
        hidden1 = self.rectify(T.dot(X, w_h))

        # Add more layers here #
        hidden2 = self.rectify(T.dot(hidden1, w_h2))  # Activation method (rectify) Layer 2

        # Hidden to output -> Cost function (softmax) Last Layer
        pyx = self.soft(hidden2, w_o)
        return pyx

    ##############################
    ##                          ##
    ##         Training         ##
    ##                          ##
    ##############################

    def training(self, epochs=20, batch=128):
        print('Training ...')
        for i in range(epochs):
            error = 0
            print('Training ... [Epoch {level}]'.format(level=i))
            for start, end in zip(range(0, len(self.trX), batch),
                                  range(batch, len(self.trX), batch)):
                # print('Training ... [Image {image}]'.format(image=start))
                error += self.train(self.trX[start:end], self.trY[start:end])
            print(error)
            solutions = np.argmax(self.teY, axis=1)
            guessed = self.predict(self.teX)
            correct = 0
            for i in range(len(guessed)):
                if guessed[i] == solutions[i]:
                    correct += 1

            print(correct)

            #print(i, str(np.mean( == )*100)+'%')

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

    def __init__(self, data, nodes=[784, 784, 10]):
        # Something with data
        self.trX, self.teX, self.trY, self.teY = mnist.mnist(onehot=True)

        ## Theano Variables ##
        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        ### Synapse Weights ###
        self.weights = []
        # Synapses and Neurons
        if len(nodes) % 1 is not 0:
            raise ValueError('Weights must be even number and come as input/output pairs')
        else:
            for i in range(1, len(nodes)):
                self.weights.append(self.gen_weights((nodes[i-1], nodes[i])))

        ### Create model ##
        ## Probability outputs and maxima predictions
        self.py_x = self.dynamic_model(self.X, self.weights)
        # index of highest in output
        self.y_x = T.argmax(self.py_x, axis=1)

        ## Classification metrix to optimize, ERROR
        self.error = T.mean(T.nnet.categorical_crossentropy(self.py_x, self.Y))

        # Cost funtion
        self.updates = self.rmsprop(self.error, self.weights)

        # Train function
        self.train = theano.function(inputs=[self.X, self.Y], outputs=self.error, updates=self.updates,
                                     allow_input_downcast=True)

        # Predict function
        self.predict = theano.function(inputs=[self.X], outputs=self.y_x, allow_input_downcast=True)

    def get_tests(self):
        return self.teX, self.teY

## BIAS

print('Starting Neural Network')
n = ANN('data')
test_x, test_y = n.get_tests()
n.training()
n.predicting([test_x[2]])
n.predicting([test_x[3]])
n.predicting([test_x[4]])
