from sklearn import datasets
import theano
from theano import tensor as T
import numpy as np
from numpy import matrix
import time
from Adaline import MineAdaline

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

import MultiLayerPerceptron


def main():
    # X, y = datasets.make_classification(1000, n_features=2, weights=[0.4, 0.6],
    #                                     n_informative=2, n_redundant=0,
    #                                     n_clusters_per_class=1, flip_y=0.01,
    #                                     shuffle=True)

    model = MineAdaline(vectorSize=2)
    X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    y = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    # y = [[1], [2], [3], [4], [5]]
    # y = [1, 2, 3, 4, 5]
    # print model.fit(X, y, 10000)
    #
    # print str(model.W.get_value())
    # print str(model.biasVisible.get_value())

    asfd = MultiLayerPerceptron.MultilayerPerceptron(topology=[2, 2], learningRate=0.0001)
    asfd.train(X, y)



if __name__ == '__main__':
    main()
