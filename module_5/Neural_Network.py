import theano
from theano import tensor as T
import numpy as np
from module_5.basics import mnist_basics as mnist


class NeuralNetwork(object):
    def floatX(self, x):
        return np.asarray(x, dtype=theano.config.floatX)

    def gen_weights(self, shape):
        return theano.shared(self.floatX(np.random.randn(*shape) * 0.01))

    ##### Activation Functions #####
    def sigmoid(self, x, w):
        return T.nnet.sigmoid(T.dot(x, w))

    def soft(self, h, w):
        return T.nnet.softmax(T.dot(h, w))

    def rectify(self, x):
        return T.maximum(x, 0.)

    # Black box of computation
    # This is where the activation functions are applied
    # Represents what happens in a neuron
    def model(self, X, w_h, w_o):
        # Input to hidden -> Activation method (sigmoid) Layer 1
        h = self.sigmoid(X, w_h)

        # Add more layers here

        # Hidden to output -> Cost function (softmax) Last Layer
        pyx = self.soft(h, w_o)
        return pyx

    ## Cost function Stochastic Gradient Decent
    def sgd(self, cost, params, lr=0.05):
        gradients = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, gradients):
            updates.append([p, p - g * lr])
        return updates

    def __init__(self):
        ### Load Data ###

        trX, teX, trY, teY = mnist.mnist(onehot=True)
        # Load train Data
        # trX, trY = mnist.load_mnist(dataset='training')
        # Load test Data
        # teX, teY = mnist.load_mnist(dataset='testing')

        ### Init Variables ###
        X = T.matrix()
        Y = T.matrix()

        ### Synapse Weights ###
        w_h = self.gen_weights((784, 625))
        w_o = self.gen_weights((625, 10))

        ### Init model ##
        py_x = self.model(X, w_h=w_h, w_o=w_o)
        y_x = T.argmax(py_x, axis=1)

        # Start
        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        params = [w_h, w_o]  # the weights

        # Evaluate updates for weights
        updates = self.sgd(cost, params)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        ### Run

        for i in range(10):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                cost = train(trX[start:end], trY[start:end])
                print(cost)
        print(np.mean(np.argmax(teY, axis=1) == predict(teX)))


ann = NeuralNetwork()
