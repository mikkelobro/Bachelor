import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Signal setup
# -----------------------------
fs = 1000
t = np.arange(0, 2, 1/fs)

# Clean signal
s = np.sin(2 * np.pi * 50 * t)

# Noise
noise = 0.5 * np.random.randn(len(t))

# Reference noise (correlated)
noise_ref = np.roll(noise, 2)

# Desired signal (signal + noise)
d = s + noise

# Input to adaptive filter
x = noise_ref

# -----------------------------
# LMS parameters
# -----------------------------
L = 32
mu = 0.01

w = np.zeros(L)
y = np.zeros(len(x))
e = np.zeros(len(x))

# Store weights over time (for plotting)
w_history = np.zeros((len(x), L))

# -----------------------------
# LMS algorithm
# -----------------------------
for n in range(L, len(x)):
    x_vec = x[n:n-L:-1]

    y[n] = np.dot(w, x_vec)     # estimated noise
    e[n] = d[n] - y[n]          # filtered signal

    w = w + mu * e[n] * x_vec

    w_history[n, :] = w

# -----------------------------
# Plot EVERYTHING
# -----------------------------
plt.figure(figsize=(12, 10))

# 1. Noisy vs filtered
plt.subplot(4,1,1)
plt.plot(t, d, label="Noisy signal d(n)", alpha=0.6)
plt.plot(t, e, label="Filtered signal e(n)")
plt.title("Before vs After Filtering")
plt.legend()
plt.grid()

# 2. Estimated noise
plt.subplot(4,1,2)
plt.plot(t, y)
plt.title("Estimated Noise y(n)")
plt.grid()

# 3. Squared error (learning)
plt.subplot(4,1,3)
plt.plot(t, e**2)
plt.title("Squared Error e(n)^2 (Convergence)")
plt.grid()

# 4. Coefficient evolution (first 5)
plt.subplot(4,1,4)
for i in range(5):
    plt.plot(t, w_history[:, i], label=f"w{i}")
plt.title("Filter Coefficient Adaptation")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()