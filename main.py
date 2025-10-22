import matplotlib.pyplot as plt

from data.data_generator import DataGenerator
from neuron.linear_neuron import LinearNeuron

data = DataGenerator(a1=100, a2=20, f1=250, f2=1000)
t, signal = data.generate()
normalized_signal = data.normalize(signal)
LNU = LinearNeuron(learning_rate=0.1, signal=normalized_signal, t=t, look_back_window=5)
LNU.learn(epochs=2)

if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label="Original Signal")
    plt.subplot(2, 1, 2)
    plt.plot(t, normalized_signal, label="Normalized Signal (Z-score)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
