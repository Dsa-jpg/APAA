import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

from src.data.data_generator import DataGenerator
from src.neuron.linear_neuron import LinearNeuron

data = DataGenerator(a1=100, a2=20, f1=250, f2=100)
t, signal = data.generate()
normalized_signal = data.normalize(signal)
LNU = LinearNeuron(learning_rate=0.001, signal=normalized_signal, t=t, look_back_window=5)
LNU.learn(epochs=100)
Y_pred = LNU.pred
Error = LNU.error
last_key = list(Y_pred.keys())[-1]
print(f"Predicted value: {Y_pred[last_key]}, Time stamp: {last_key}")
predicted_values = [np.nan] * LNU.look_back_window + list(Y_pred.values())
weights_history = np.array(LNU.weights_history)

if __name__ == '__main__':
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, normalized_signal, color='red', label="Normalized Signal")
    plt.scatter(t, predicted_values, s=5, label="Predicted Values",
                marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(LNU.error) + 1), LNU.error, color='blue', label="Error")
    plt.title("Error Progress During Learning")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    for i in range(weights_history.shape[1]):
        plt.plot(range(1, len(weights_history) + 1), weights_history[:, i], label=f'Weight {i + 1}')
    plt.title("Weight progress during learning")
    plt.xlabel("Time [s]")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
