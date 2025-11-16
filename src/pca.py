import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

matplotlib.use('TkAgg')

# --- Syntetický signál ---
n_weights = 10
a1, a2, a3, a4 = 4, -2, 2, -1
f, f2, f3 = 130, 250, 50
t_stop = 10
fs = 2000
dt = 1 / fs

t = np.arange(0, t_stop, dt)
N = len(t)

y = a1 * np.sin(2 * np.pi * f * t) + a2 * np.sin(2 * np.pi * f * t)

t1, t2 = int(len(t) * 0.2), int(len(t) * 0.4)
t3, t4 = int(len(t) * 0.7), int(len(t) * 0.8)

y[t1:t2] += a3 * np.sin(2 * np.pi * f2 * t[t1:t2])
y[t3:t4] += a3 * np.cos(2 * np.pi * f3 * t[t3:t4])

# --- Lagged matrix ---
X = np.array([y[k - n_weights:k] for k in range(n_weights, N)])

# --- PCA přes korelační matici ---
X_centered = X - np.mean(X, axis=0)
corr_X = np.corrcoef(X_centered.T)

d, v = np.linalg.eigh(corr_X)
idx = np.argsort(d)[::-1]
d = d[idx]
v = v[:, idx]

m = 2  # počet PCA komponent
top_vectors = v[:, :m]
X_pca = (X_centered @ top_vectors) @ v[:, :m].T

mu = 0.01

w = np.random.randn(n_weights) / n_weights  # np.zeros(n_weights)
y_pred = np.zeros(N)
error = np.zeros(N)
wall = np.zeros((N, n_weights))
delta_wall = np.zeros((N, n_weights))

for k in range(n_weights, N):
    x = X_pca[k - n_weights]
    y_pred[k] = np.dot(w, x)
    error[k] = y[k] - y_pred[k]
    dw = mu * error[k] * x
    w += dw
    wall[k] = w
    delta_wall[k] = dw

fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

# --- 1. Originální signál a predikce ---
axes[0].plot(t, y, label="y (original signal)", color='red')
axes[0].plot(t, y_pred, label="ŷ (prediction)", color='green')
axes[0].set_title("Signal y and its Prediction")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc='upper right')
axes[0].grid(True)

# --- 2. Chyba predikce (squared error) ---
axes[1].plot(t, error ** 2, label="Squared Error", color='blue')
axes[1].set_title("Prediction Error e[k]")
axes[1].set_ylabel("Error")
axes[1].legend(loc='upper right')
axes[1].grid(True)

# --- 3. Vývoj vah ---
for i in range(n_weights):
    axes[2].plot(t, wall[:, i], label=f"w{i + 1}")
axes[2].set_title("Evolution of Weights w[k]")
axes[2].set_ylabel("Weight Value")
axes[2].grid(True)

# --- 4. Vývoj změny vah ---
axes[3].plot(t, np.sum(delta_wall ** 2, axis=1), color='purple', label="Σ(Δw²)")
axes[3].set_title("Evolution of Weight Change (Δw)")
axes[3].set_xlabel("Samples [k]")
axes[3].set_ylabel("Sum of Δw²")
axes[3].grid(True)

# --- Nastavení osy X ---
axes[3].xaxis.set_major_locator(ticker.MultipleLocator(10))
axes[3].xaxis.set_minor_locator(ticker.MultipleLocator(5))
axes[3].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.tight_layout(pad=2.5)
plt.show()

X_pca_simple = X_centered @ top_vectors  # délka (N-n_weights, 2)
plt.figure(figsize=(6, 6))
plt.scatter(X_pca_simple[:, 0], X_pca_simple[:, 1], alpha=0.7)
plt.xlabel("c1 (1st PCA component)")
plt.ylabel("c2 (2nd PCA component)")
plt.title("First two PCA components (c1 vs c2)")
plt.grid(True)
plt.axis('equal')
plt.show()

# --- Eigenvalues ---
plt.figure(figsize=(8, 4))
plt.plot(abs(d), 'o-', color='red')
plt.title("Eigenvalues of correlation matrix")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()
