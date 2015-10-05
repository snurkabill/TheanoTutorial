from sklearn import datasets
import theano
from theano import tensor as T
import numpy as np
from numpy import matrix

from Adaline import MineAdaline

def main():

    model = MineAdaline(vectorSize=2)
    X = [[1, 1], [2, 2], [3, 3]]
    y = [1, 2, 3]
    print model.fit(X, y, 10000)

if __name__ == '__main__':
    main()
