import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib import ticker

matplotlib.use('TkAgg')

# -------------------------------------#
record = wfdb.rdrecord('./nsrdb/16265')
annotation = wfdb.rdann('./nsrdb/16265', 'atr')

y = record.p_signal[:, 0]  # first channel
print(y)

# Z-scores
y_z = (y - np.mean(y)) / np.std(y)
y = y_z
# Z-scores

N = len(y)
fs = int(record.fs)
t = np.arange(N) / fs

# Signal limit _s
t_max = 0.95
N_max = int(fs * t_max)

####### Segmentace a posun dat  #########
segment = y[:N_max]
y = [segment] * 250 + [segment + 1] + [segment] * 250

y = np.concatenate(y)
t = np.arange(len(y)) / fs
##############

N = len(y)
# -------------------------------------#

mu = 0.01
n_weights = 125
# a1 = 4
# a2 = -2
# a3 = 2
# a4 = -1
# f = 130
# f2 = 250
# f3 = 50
# t_stop = 10
# fs = 2000
# dt = 1 / fs
#
# t = np.arange(0, t_stop, dt)
# N = len(t)
#
# y = a1 * np.sin(2 * np.pi * f * t) + a2 * np.sin(2 * np.pi * f * t)
#
# t1 = int(len(t) * 0.2)
# t2 = int(len(t) * 0.4)
# t3 = int(len(t) * 0.7)
# t4 = int(len(t) * 0.8)
#
# y[t1:t2] += a3 * np.sin(2 * np.pi * f2 * t[t1:t2])
# y[t3:t4] += a3 * np.cos(2 * np.pi * f3 * t[t3:t4])


# y = record.p_signal
# fs = int(record.fs)
#
# t_max = 10
# N_max = int(fs * t_max)
# y = y[:N_max, :]

w = np.random.randn(n_weights) / n_weights  # np.zeros(n_weights)
y_pred = np.zeros(N)
error = np.zeros(N)
wall = np.zeros((N, n_weights))
delta_wall = np.zeros((N, n_weights))

# Convariance #
# X = np.array([y[k - n_weights:k] for k in range(n_weights, N)])
# print(f"The matrix {X}")
# m = 125
# cov_y = np.corrcoef(X.T)
#
# d, v = np.linalg.eig(cov_y)
# # print(d)
# # print(v)
# #
# c = cov_y @ v[:, :m]
# # print(c)
# xpca = c @ v.T[:, :m]
# print(f"The matrix PCA {xpca}")
# print(xpca.shape)

# Convariance #
for k in range(n_weights, N):
    x = y[k - n_weights:k]
    y_pred[k] = np.dot(w, x)
    error[k] = y[k] - y_pred[k]
    dw = mu * error[k] * x
    w = w + dw
    wall[k] = w
    delta_wall[k] = dw

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

for i in range(n_weights):
    axes[2].plot(t, wall[:, i], label=f"w{i + 1}")
axes[2].set_title("Evolution of Weights w[k]")
axes[2].set_ylabel("Weight Value")
# axes[2].legend(loc='upper right', ncol=2, fontsize='small')
axes[2].grid(True)

axes[3].plot(t, np.sum(delta_wall ** 2, axis=1), color='purple', label="Σ(Δw²)")
axes[3].set_title("Evolution of Weight Change (Δw)")
axes[3].set_xlabel("Samples [k]")
axes[3].set_ylabel("Sum of Δw²")
# axes[3].legend(loc='upper right', fontsize='small')
axes[3].grid(True)

axes[3].xaxis.set_major_locator(ticker.MultipleLocator(10))  # major ticks every 1 second
axes[3].xaxis.set_minor_locator(ticker.MultipleLocator(5))  # minor ticks every 0.1 s
axes[3].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.tight_layout(pad=2.5)
plt.show()
