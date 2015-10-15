
__author__ = 'snurkabill'

import theano
import numpy as np
from theano import tensor as T

theanoFloat = theano.config.floatX


class MultilayerPerceptron:
    def __init__(self, topology, learningRate=0.01, momentum = 0.1, initialWeights=None,
                 initialBiases=None):
        self.topology = topology
        self.numberOfLayers = len(topology)
        self.learningRate = learningRate
        self.momentum = momentum
        # self.weights = []
        # self.biases = []
        #
        # for i in xrange(self.numberOfLayers - 1):
        #     self.weights.append(np.asarray(np.random.normal(0, 0.01,
        #                            (self.topology[i], self.topology[i+1])),
        #                           dtype=theanoFloat))
        #     self.biases.append(np.asarray(np.zeros(self.topology[i + 1])))
        #
        # self.weights = theano.shared(value=np.asarray(self.weights, dtype=theanoFloat),
        #                              name='W')
        # self.biases = theano.shared(value=np.asarray(self.biases,dtype=theanoFloat),
        #                             name='bvis')
        initWeights = np.asarray(np.random.uniform(
                      low=-4 * np.sqrt(6. / (topology[0] + topology[1])),
                      high=4 * np.sqrt(6. / (topology[0] + topology[1])),
                      size=(topology[0], topology[1])), dtype=theanoFloat)

        # initWeights = np.asarray(np.random.uniform(
        #               low=-4 * np.sqrt(6. / (1 + topology[0])),
        #               high=4 * np.sqrt(6. / (1 + topology[0])),
        #               size=topology[0]), dtype=theano.config.floatX)
        self.weights = theano.shared(value=np.asarray(initWeights,
                                  dtype=theano.config.floatX),
                        name='W')
        self.biases = theano.shared(np.zeros(topology[1], dtype=theanoFloat))
        # self.biases = theano.shared(0., name='bvis')

    def train(self, data, labels, numberOfEpochs=10000):
        data = np.array(data, dtype=theanoFloat)
        labels = np.array(labels, dtype=theanoFloat)

        print "data shape" + str(data.shape)
        print "data labels" + str(labels.shape)
        print "data weighs" + str(self.weights.shape)
        print "data biases" + str(self.biases.shape)

        # The mini-batch data is a matrix
        X = T.matrix(name='X', dtype=theanoFloat)
        # labels[start:end] this needs to be a matrix because we output probabilities
        Y = T.matrix(name='y', dtype=theanoFloat)

        potential = T.dot(X, self.weights) + self.biases
        activation = potential
        cost = T.sum((activation - Y) ** 2)
        gw, gb = T.grad(cost, [self.weights, self.biases])
        train = theano.function(inputs=[X, Y], outputs=[activation, cost],
                                updates=
                                [[self.weights, self.weights - self.learningRate * gw],
                                 [self.biases, self.biases - self.learningRate * gb]],
                                name="optimize")
        theano.printing.pydotprint(cost, outfile="cost.png", var_with_name_simple=True)
        for i in xrange(numberOfEpochs):
            propagatedValues, error = train(data, labels)
            print "Error: " + str(error)
