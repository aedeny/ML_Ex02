import sys

import numpy as np


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


def cross_entropy(yhat, y):
    return - np.sum(y * np.log(yhat + 1e-6))


def sgd(params, learning_rate):
    for param in params:
        param[:] = param - learning_rate * param.grad


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.w = np.zeros((n_in, n_out))  # initialize w 0
        self.b = np.zeros(n_out)  # initialize bias 0
        self.y_hat = None

    def train(self, lr=0.1, input=None, l2_reg=0.00):
        if input is not None:
            self.x = input

        # Shuffles arrays
        s = np.arange(self.x.shape[0])
        np.random.shuffle(s)
        self.x = self.x[s]
        self.y = self.y[s]

        for x, y in zip(self.x, self.y):
            z = np.dot(x, self.w) + self.b
            self.y_hat = np.argmax(softmax(z)) + 1
            if y != self.y_hat:
                for i in range(1, 4):
                    if i == y:
                        loss_dw = softmax(z) * x - x
                        loss_db = softmax(z) - 1
                    else:
                        loss_dw = softmax(z) * x
                        loss_db = softmax(z)

                    self.w[i-1] -= lr * loss_dw
                    self.b[i-1] -= lr * loss_db

    def negative_log_likelihood(self):
        return - np.sum(self.y * np.log(self.y_hat))

    def predict(self, x):
        return softmax(np.dot(x, self.w) + self.b)


if __name__ == '__main__':
    # Creates data
    training_data = []
    x = np.empty((0, 1), float)
    y = np.empty((0, 1), int)
    for a in [1, 2, 3]:
        for i in range(100):
            x = np.append(x, np.array([np.random.normal(2 * a, 1, 1)]), axis=0)
            y = np.append(y, np.array([[a]]), axis=0)

    # Constructs the Logistic Regression classifier
    classifier = LogisticRegression(x, y, 1, 3)

    # Trains model
    learning_rate = 0.01
    n_epochs = 200

    for epoch in xrange(n_epochs):
        classifier.train(learning_rate)
        cost = classifier.negative_log_likelihood()
        print("Training epoch " + str(epoch) + " cost is " + str(cost))
        learning_rate *= 0.95

    # Tests model
    x = np.array([3.23149])
    print >> sys.stderr, classifier.predict(x)
