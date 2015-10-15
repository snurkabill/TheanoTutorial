from CostFunctions import LeastSquares

__author__ = 'snurkabill'

import theano
import numpy as np
from theano import tensor as T
from ActivationFunctions import Sigmoid, Identity

theanoFloat = theano.config.floatX


class MultilayerPerceptron:
    def __init__(self, topology, activationFunction=Sigmoid(), lastActivationFunction=Identity(),
                 costFunction=LeastSquares(), learningRate=0.01, momentum=0.1):
        self.topology = topology
        self.numberOfLayers = len(topology)
        self.learningRate = learningRate
        self.momentum = momentum
        self.activationFunction=activationFunction
        self.lastActivationFunction = lastActivationFunction
        self.costFunction = costFunction

        initWeights = []
        initBiases = []

        for i in xrange(len(topology) - 1):
            initWeights.append(np.asarray(np.random.uniform(
                      low=-4 * np.sqrt(6. / (topology[i] + topology[i + 1])),
                      high=4 * np.sqrt(6. / (topology[i] + topology[i + 1])),
                      size=(topology[i], topology[i + 1])), dtype=theanoFloat))

            initBiases.append(np.zeros(topology[i + 1], dtype=theanoFloat))

        self.weights = theano.shared(value=np.asarray(initWeights, dtype=theanoFloat), name='W')
        self.biases = theano.shared(value=np.asarray(initBiases), name="B")
        # self.biases = theano.shared(0., name='bvis')

    def train(self, data, labels, numberOfEpochs=10000):
        data = np.array(data, dtype=theanoFloat)
        labels = np.array(labels, dtype=theanoFloat)
        dataSymbolic = T.matrix(name='data', dtype=theanoFloat)
        labelsSymbolic = T.matrix(name='labels', dtype=theanoFloat)

        propagatedData = dataSymbolic
        potential = T.dot(propagatedData, self.weights[0]) + self.biases[0]
        activation = potential
        cost = T.sum(self.costFunction.cost(activation, labelsSymbolic))
        gw, gb = T.grad(cost, [self.weights, self.biases])
        train = theano.function(inputs=[dataSymbolic, labelsSymbolic], outputs=[activation, cost],
                                updates=
                                [[self.weights, self.weights - self.learningRate * gw],
                                 [self.biases, self.biases - self.learningRate * gb]],
                                name="optimize")
        theano.printing.pydotprint(cost, outfile="cost.png", var_with_name_simple=True)
        for i in xrange(numberOfEpochs):
            propagatedValues, error = train(data, labels)
            print "Error: " + str(error)
