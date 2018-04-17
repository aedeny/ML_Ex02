import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from matplotlib.legend_handler import HandlerLine2D


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


class LogisticRegression(object):
    def __init__(self, in_data, label, n_in, n_out):
        self.x = in_data
        self.y = label
        self.w = np.zeros((n_in, n_out))  # initialize w 0
        self.b = np.zeros((n_in, n_out))  # initialize bias 0

    def train(self, num_of_epochs=30, lr=0.1, in_data=None):
        if in_data is not None:
            self.x = input

        for epoch in range(num_of_epochs):

            # Shuffles arrays
            s = np.arange(self.x.shape[0])
            np.random.shuffle(s)
            self.x = self.x[s]
            self.y = self.y[s]

            for x, y in zip(self.x, self.y):
                z = np.dot(x, self.w) + self.b

                # Predicts
                current_softmax = softmax(z)
                y_hat = np.argmax(current_softmax) + 1
                if y != y_hat:
                    for a in range(3):
                        if a + 1 == y:
                            self.w[0, a] -= lr * (current_softmax[0, a] * x - x)
                            self.b[0, a] -= lr * (current_softmax[0, a] - 1)
                        else:
                            self.w[0, a] -= lr * current_softmax[0, a] * x
                            self.b[0, a] -= lr * current_softmax[0, a]

    def predict(self, x_val):
        return np.argmax(softmax(np.dot(x_val, self.w) + self.b)) + 1

    def draw_plot(self):
        x_values = np.arange(0, 10, 0.1).tolist()
        pdf_y_values = []
        model_y_values = []

        # Creates y values
        for val in x_values:
            pdf_y_values.append(sc.norm(2, 1).pdf(val) / (sc.norm(2, 1).pdf(val) + sc.norm(4, 1).pdf(val) +
                                                          sc.norm(6, 1).pdf(val)))
            model_y_values.append(softmax(np.dot(self.w, val) + self.b).tolist()[0][0])

        # Sets graph's settings
        fig = plt.figure(0)
        fig.canvas.set_window_title('Normal Distribution vs. Logistic Regression')
        plt.title("Normal Distribution vs. Logistic Regression")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.axis([0, 10, 0, 1.5])
        plt.xlabel('x')
        plt.ylabel('Probability')
        normal_graph, = plt.plot(x_values, pdf_y_values, 'r--', label="Normal Distribution")
        plt.plot(x_values, model_y_values, label="Logistic Regression")
        plt.legend(handler_map={normal_graph: HandlerLine2D(numpoints=4)})

        plt.show()


if __name__ == '__main__':
    # Creates data
    training_data = []
    x = np.empty((0, 1), float)
    y = np.empty((0, 1), int)
    for a in [1, 2, 3]:
        for i in range(100):
            x = np.append(x, np.array([np.random.normal(2 * a, 1, 1)]), axis=0)
            y = np.append(y, np.array([[a]]), axis=0)

    # # Draws a scatter of points
    # plt.scatter(x[0:100], [1] * 100, color='green')
    # plt.scatter(x[100:200], [2] * 100, color='red')
    # plt.scatter(x[200:300], [3] * 100, color='blue')
    # plt.show()

    # Constructs the Logistic Regression classifier
    classifier = LogisticRegression(x, y, 1, 3)

    classifier.train()
    classifier.draw_plot()
