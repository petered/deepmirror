import numpy as np
from theano.tensor.nnet import sigm
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


class NearestNeighbour:

    def __init__(self, metric = 'L2'):

        self.distance_func = {
            'L2': lambda u, v: ((u[:, None, :] - v[None, :, :])**2).sum(axis=2),
            'L1': lambda u, v: np.abs(u[:, None, :] - v[None, :, :]).sum(axis=2),
            'dot': lambda u, v: -np.einsum('mi,nj->mn', u, v),
            }
        assert metric in ('L2', 'L1', 'dot')

    def fit(self, x, y):
        """
        :param x: An (n_training_samples, n_dims) array
        :param y: An (n_training_samples, ) array of labels.
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """
        :param x:  An (n_training_samples, n_dims) array
        :return: An (n_training_samples, ) array of labels.
        """
        distances_sq = ((self.x[:, None, :] - x[None, :, :])**2).sum(axis=2)  # (n_training_samples, n_test_samples)
        nearset_neighbours = self.y[np.argmin(distances_sq, axis=0)]  # (n_test_samples, )
        return nearset_neighbours


class MultinomialRegression:

    def __init__(self, n_in, n_out, learning_rate = 0.1):
        self.w = np.zeros((n_in, n_out))
        self.learning_rate = learning_rate

    def train(self, x, y):
        """
        # Update the parameters of the regressor

        :param x: An (n_samples, n_dims) input
        :param y: An (n_samples, ) array of integer labels
        """
        p = softmax(x.dot(self.w))
        y_onehot = OneHotEncoding(n_classes = p.shape[1])(y)
        d_loss_d_w = np.einsum('si,sj->ij', x, (p-y_onehot)) # / x.shape[0]  # (n_in, n_out).  The mean gradient
        self.w -= self.learning_rate * d_loss_d_w

    def predict(self, x):
        p = softmax(x.dot(self.w))
        return p.argmax(axis=1)  # Obviously the softmax isn't necessary, but just for the sake of illustration it's here.


class LogisticRegression:

    def __init__(self, n_in, n_out, learning_rate = 0.1):
        self.w = np.zeros((n_in, n_out))
        self.learning_rate = learning_rate

    def train(self, x, y):
        """
        # Update the parameters of the regressor

        :param x: An (n_samples, n_dims) input
        :param y: An (n_samples, ) array of integer labels
        """
        p = sigm(x.dot(self.w))
        d_loss_d_w = np.einsum('si,sj->ij', x, p-y) # / x.shape[0]  # (n_in, n_out).  The mean gradient
        self.w -= self.learning_rate * d_loss_d_w

    def predict(self, x):
        p = softmax(x.dot(self.w))
        return p.argmax(axis=1)  # Obviously the softmax isn't necessary, but just for the sake of illustration it's here.



def softmax(x):
    """
    :param x: An (n_samples, n_dims) array
    :return: An (n_samples, n_dims) array of output probabilities
    """
    u = np.exp(x)
    out = u / u.sum(axis=1, keepdims=True)
    return out