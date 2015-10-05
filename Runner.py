from sklearn import datasets
import theano
from theano import tensor as T
import numpy as np
from numpy import matrix

from Perceptron import VahysPerceptron

def main():
    #
    # X, y = datasets.make_classification(1000, n_features=2, weights=[0.4,0.6],
    #                                 n_informative=2, n_redundant=0,
    #                                 n_clusters_per_class=1, flip_y=0.01,
    #                                 shuffle=True)
    # X2 = T.dmatrix("X")
    # y2 = T.dvector("Y")

    # X2.tag.test_value = X
    # y2.tag.test_value = y
    model = VahysPerceptron(vectorSize=2)
    X = [[1, 1], [2, 2], [3, 3]]
    y = [1, 2, 3]
    print model.fit(X, y, 10000)



if __name__ == '__main__':
    main()
