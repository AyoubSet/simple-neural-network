import numpy as np

class Utils:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def likelihood(output, y_train):
        output = np.array(output)
        y_train = np.array(y_train)
        eps = 1e-9
        output = np.clip(output, eps, 1 - eps)

        return np.prod(output**y_train * (1 - output)**(1 - y_train))

    @staticmethod
    def log_likelihood(output, y_train):
        output = np.array(output)
        y_train = np.array(y_train)
        eps = 1e-9  # small constant
        output = np.clip(output, eps, 1 - eps)

        return np.sum(
            y_train * np.log(output) +
            (1 - y_train) * np.log(1 - output)
        )
        
    @staticmethod
    def nll(output, y_train):
        output = np.array(output)
        y_train = np.array(y_train)
        eps = 1e-9  # small constant
        output = np.clip(output, eps, 1 - eps)

        return -np.sum(
            y_train * np.log(output) +
            (1 - y_train) * np.log(1 - output)
        )

