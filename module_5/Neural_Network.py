import theano
from theano import tensor as T
import numpy as np
from module_5.basics import mnist_basics as mnist


class NeuralNetwork(object):
    def gen_weights(self, shape):
        return theano.shared(np.random.randn(*shape) * 0.01)

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
    def model(self, X, w_h, w_h2, w_o):
        print(w_h)
        # Input to hidden -> Activation method (sigmoid) Layer 1
        hidden1 = self.rectify(T.dot(X, w_h))

        # Add more layers here #
        hidden2 = self.rectify(T.dot(hidden1, w_h2))  # Activation method (rectify) Layer 2

        # Hidden to output -> Cost function (softmax) Last Layer
        pyx = self.soft(hidden2, w_o)
        return pyx

    ## Cost function Stochastic Gradient Decent
    ## This is the function to minimize
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
        ### number of input to number of output
        weight_hidden = self.gen_weights((784, 625))
        weight_hidden2 = self.gen_weights((625, 625))
        weight_output = self.gen_weights((625, 10))

        ### Init model ##
        ## Probability outputs and maxima predictions
        py_x = self.model(X, w_h=weight_hidden, w_h2=weight_hidden2, w_o=weight_output)
        y_x = T.argmax(py_x, axis=1)

        ## Classification metrix to optimize
        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        params = [weight_hidden, weight_hidden2, weight_output]  # the weights

        # Cost funtion
        updates = self.sgd(cost, params)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        ### Run
        epochs = 20
        number_from_each_class = 128
        for i in range(epochs):

            for start, end in zip(range(0, len(trX), number_from_each_class),
                                  range(number_from_each_class, len(trX), number_from_each_class)):
                cost = train(trX[start:end], trY[start:end])

            print(u" ")
            print(u"Accuracy:")
            print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)))



ann = NeuralNetwork()
