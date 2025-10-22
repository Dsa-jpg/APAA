import time

import numpy as np


class LinearNeuron(object):

    def __init__(self, learning_rate: float, signal, t, look_back_window):
        self.learning_rate = learning_rate
        self.signal = signal
        self.t = t
        self.look_back_window = look_back_window
        # Generates random weights in interval [-1,1]
        self.weights = np.random.random(self.look_back_window) * 2 - 1

    def _get_error_for_sample(self, measured_signal, predicted_signal):
        """
        Does substitution of  measured_signal and predicted_signal.

        :param measured_signal:
        :param predicted_signal:
        :return: error_for_sample
        """
        return measured_signal - predicted_signal

    def _update_weights(self, e, x_vector):
        """
        Calculates the new weights based on the error and x_vector.
        Using backpropagation.

        :param e:
        :param x_vector:
        :return: None
        """
        self.weights += self.learning_rate * e * x_vector

    def _predict(self, x_vector):
        """
        Sum of input of x_vector * weights.

        :param x_vector:
        :return:
        """
        return np.dot(x_vector, self.weights)

    def learn(self, epochs=10, debug=False):
        for epoch in range(epochs):
            total_error = 0

            for k in range(self.look_back_window, len(self.signal)):
                x_vec = self.signal[k - self.look_back_window:k]  # takes last n samples before k for neurons output
                y = self.signal[k]  # value that is being predicted by LNU

                y_predict = self._predict(x_vec)
                error = self._get_error_for_sample(y, y_predict)

                self._update_weights(error, x_vec)
                total_error += error ** 2
                if debug:
                    print(f"predicted_value: {y_predict}, actual_value: {y} error: {error}")
                    time.sleep(self.learning_rate)

            print(f"epoch: {epoch + 1}, total_error: {total_error:.4f}")
