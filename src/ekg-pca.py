import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib import ticker

matplotlib.use('TkAgg')

record = wfdb.rdrecord('./src/nsrdb/16265')
annotation = wfdb.rdann('./src/nsrdb/16265', 'atr')

y = record.p_signal[:, 0]
y = (y - np.mean(y)) / np.std(y)

N = len(y)
fs = int(record.fs)
t = np.arange(N) / fs

t_max = 0.95
N_max = int(fs * t_max)
segment = y[:N_max]
y = np.concatenate([segment] * 250 + [segment + 1] + [segment] * 250)
t = np.arange(len(y)) / fs
N = len(y)

mu = 0.01
n_weights = 125

X = np.array([y[k - n_weights:k] for k in range(n_weights, N)])

X_centered = X - np.mean(X, axis=0)
corr_X = np.corrcoef(X_centered.T)
d, v = np.linalg.eigh(corr_X)
idx = np.argsort(d)[::-1]
d = d[idx]
v = v[:, idx]

m = 40
top_vectors = v[:, :m]
X_pca = (X_centered @ top_vectors) @ v[:, :m].T

# --- LNU trénink ---
w = np.random.randn(n_weights) / n_weights
y_pred = np.zeros(N)
error = np.zeros(N)
wall = np.zeros((N, n_weights))
delta_wall = np.zeros((N, n_weights))

for k in range(n_weights, N):
    x = X_pca[k - n_weights]  # délka m
    y_pred[k] = np.dot(w, x)
    error[k] = y[k] - y_pred[k]
    dw = mu * error[k] * x
    w += dw
    wall[k] = w
    delta_wall[k] = dw

# --- Grafy LNU ---
fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

axes[0].plot(t, y, label="y (original signal)", color='red')
axes[0].plot(t, y_pred, label="ŷ (prediction)", color='green')
axes[0].set_title("Signal y and its Prediction")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc='upper right')
axes[0].grid(True)

axes[1].plot(t, error ** 2, label="Squared Error", color='blue')
axes[1].set_title("Prediction Error e[k]")
axes[1].set_ylabel("Error")
axes[1].legend(loc='upper right')
axes[1].grid(True)

for i in range(m):
    axes[2].plot(t, wall[:, i], label=f"w{i + 1}")
axes[2].set_title("Evolution of Weights w[k]")
axes[2].set_ylabel("Weight Value")
axes[2].grid(True)

axes[3].plot(t, np.sum(delta_wall ** 2, axis=1), color='purple', label="Σ(Δw²)")
axes[3].set_title("Evolution of Weight Change (Δw)")
axes[3].set_xlabel("Samples [k]")
axes[3].set_ylabel("Sum of Δw²")
axes[3].grid(True)

axes[3].xaxis.set_major_locator(ticker.MultipleLocator(10))
axes[3].xaxis.set_minor_locator(ticker.MultipleLocator(5))
axes[3].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.tight_layout(pad=2.5)
plt.show()

# --- Scatter prvních dvou PCA komponent ---
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel("c1 (1st PCA component)")
plt.ylabel("c2 (2nd PCA component)")
plt.title("First two PCA components (c1 vs c2)")
plt.grid(True)
plt.axis('equal')
plt.show()

# --- Eigenvalues ---
plt.figure(figsize=(8, 4))
plt.plot(d, 'o-', color='red')
plt.title("Eigenvalues of correlation matrix")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()
