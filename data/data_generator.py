import numpy as np


class DataGenerator(object):

    def __init__(self, a1: float, a2: float, f1: float, f2: float, duration: float = 100.0,
                 sampling_rate: float = 1000):
        self.a1 = a1
        self.a2 = a2
        self.f1 = f1
        self.f2 = f2
        self.duration = duration
        self.sampling_rate = sampling_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate))
        signal = self.a1 * np.sin(self.a1 * np.pi * self.f1 * t) + self.a2 * np.sin(self.a2 * np.pi * self.f2 * t)
        return t, signal

    @staticmethod
    def normalize(signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std
