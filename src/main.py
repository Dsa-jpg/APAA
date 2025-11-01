import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

a = 4
a2 = -2
f = 130
t_stop = 10
dt = 1 / 2000
mu = 0.01
ny = 2
nw = ny
nx = ny

t = np.arange(0, t_stop, dt)
y = a * np.sin(2 * np.pi * f * t) + a2 * np.sin(2 * np.pi * f * t)

N = len(t)
w = np.random.randn(nw) / nw  # np.zeros(nw)
yn = np.zeros(N)
e = np.zeros(N)
wall = np.zeros((N, nw))
dwall = np.zeros((N, nw))

for k in range(ny, N):
    x = y[k - ny:k]
    yn[k] = np.dot(w, x)
    e[k] = y[k] - yn[k]
    dw = mu * e[k] * x
    w = w + dw
    wall[k] = w
    dwall[k] = dw

plt.figure(figsize=(8, 8))

plt.subplot(5, 1, 1)
plt.plot(t, y, label="y (signal)", color='red')
plt.plot(t, yn, label="ŷ (predikce)", color='green')
plt.title("Signál y")
plt.xlabel("k samples")
plt.ylabel("Amplituda")
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(e ** 2, label="Error", color='blue')
plt.title("Chyba e[k]")
plt.xlabel("k samples")
plt.ylabel("Chyba")
plt.grid(True)

plt.subplot(5, 1, 3)
for i in range(nw):
    plt.plot(t, wall[:, i], label=f"w{i + 1}")
plt.title("Weights w")
plt.xlabel("k samples")
plt.ylabel("w")

plt.grid(True)

plt.show()
