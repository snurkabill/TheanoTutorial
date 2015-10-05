__author__ = 'snurkabill'

import theano
import numpy as np
from numpy import random
from theano import tensor as T

class MineAdaline:

    def __init__(self, vectorSize, learningRate=0.01):
        self.learningRate = learningRate
        self.vectorSize = vectorSize
        initWeights = np.asarray(np.random.uniform(
                      low=-4 * np.sqrt(6. / (1 + vectorSize)),
                      high=4 * np.sqrt(6. / (1 + vectorSize)),
                      size=vectorSize), dtype=theano.config.floatX)
        self.W = theano.shared(value=np.asarray(initWeights,
                                  dtype=theano.config.floatX),
                        name='W')
        self.biasVisible = theano.shared(0., name='bvis')

    def fit(self, data, labels, numOfIterations=1):
        data = np.array(data, dtype=theano.config.floatX)
        labels = np.array(labels, dtype=theano.config.floatX)
        X = T.dmatrix(name='X')
        y = T.dvector(name='y')
        potential = T.dot(X, self.W) + self.biasVisible
        activation = potential
        cost = T.sum((activation - y) ** 2)
        gw, gb = T.grad(cost, [self.W, self.biasVisible])
        train = theano.function(inputs=[X, y], outputs=[activation, cost],
                                updates=
                                [[self.W, self.W - 0.01 * gw], [self.biasVisible, self.biasVisible - 0.01 * gb]],
                                name="asdf")

        for i in range(numOfIterations):
            predictions, error = train(data, labels)
            print "predictions: " + str(predictions) + " Error: " + str(error)

        theano.printing.pydotprint(potential, outfile="logreg_pydotprint_potential.png", var_with_name_simple=True)
        theano.printing.pydotprint(cost, outfile="logreg_pydotprint_cost.png", var_with_name_simple=True)
        theano.printing.pydotprint(train, outfile="logreg_pydotprint_train.png", var_with_name_simple=True)
        return train(data, labels)

