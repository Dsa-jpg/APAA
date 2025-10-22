import numpy as np


class LinearNeuron(object):

    def __init__(self, learning_rate: float, signal, t, look_back_window):
        self.learning_rate = learning_rate
        self.weights = np.random.random(signal.shape) * 2 - 1
        self.signal = signal
        self.t = t
        self.look_back_window = look_back_window

        ...

    def learn(self):
        ...

    def _get_error_for_sample(self, measured_signal, predicted_signal, time):
        ...

    ...
