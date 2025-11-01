import numpy as np


class LinearNeuron(object):

    def __init__(self, learning_rate: float, signal, t, look_back_window):
        self.learning_rate = learning_rate
        self.signal = signal
        self.t = t
        self.N = len(self.t)
        self.error = np.zeros(self.N)
        self.look_back_window = look_back_window
        self.nw = self.look_back_window
        self.nx = self.look_back_window
        self.yn = np.zeros(self.N)
        self.weights = np.random.randn(self.look_back_window) / look_back_window
        self.wall = np.zeros((self.N, self.nw))
        self.awall = np.zeros((self.N, self.nw))

    def _predict(self, x_vector):
        return np.dot(x_vector, self.weights)

    def learn(self):
        for k in range(self.look_back_window, self.N):
            x_vec = self.signal[k - self.look_back_window:k]  # takes last n samples before k for neurons output
            self.yn[k] = self._predict(x_vec)  # value that is being predicted by LNU
            error = self.signal[k] - self.yn[k]
            dw = self.learning_rate * error * x_vec
            self.weights += dw
            self.wall[k] = self.weights
            self.awall[k] = dw
