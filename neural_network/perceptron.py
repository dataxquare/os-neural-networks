from sklearn import datasets
import numpy as np

class Perceptron:
    def __init__(self, lr:float=0.01):
        self.lr = lr

    def step_func(z: int):
        return 1.0 if (z > 0) else 0.0

    def perceptron(X, y, epochs: int = 100):
        m, n = X.shape
        theta = np.zeros((n+1, 1))

        n_miss_list = []

        for epoch in range(epochs):
            n_miss = 0

            for idx, x_i in enumerate(X):
                x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

                y_hat = step_func(np.dot(x_i.T, theta))

                if (np.squeeze(y_hat) - y[idx]) != 0:
                    theta += self.lr*((y[idx] - y_hat)*x_i)
                    n_miss += 1

            n_miss_list.append(n_miss)

        return theta, n_miss_list
