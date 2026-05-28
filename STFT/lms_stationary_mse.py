import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# --------------------------------------------------
# Parameters
# --------------------------------------------------

fs = 1000
T = 2
N = int(fs * T)

L = 10

# --------------------------------------------------
# Signals
# --------------------------------------------------

n = np.arange(N)
t = n / fs

# Clean chirp signal
f0 = 20
f1 = 200

s = chirp(t, f0=f0, f1=f1, t1=T, method="linear")

# Stationary noise
noise = 1.5*np.random.randn(N)

# Desired signal
d = s + noise

# Reference noise
x = noise + 0.1*np.random.randn(N)

# --------------------------------------------------
# Calculate maximum mu
# --------------------------------------------------

Px = np.mean(x**2)

mu_max = 2 / (L * Px)

print(f"Px = {Px:.4f}")
print(f"Maximum mu = {mu_max:.4f}")

# --------------------------------------------------
# Test different mu values
# --------------------------------------------------

mu_values = [
    0.01*mu_max,
    0.05*mu_max,
    0.1*mu_max,
    0.5*mu_max,
    0.99*mu_max
]

# --------------------------------------------------
# LMS function
# --------------------------------------------------

def lms_filter(x, d, mu, L):
    N = len(d)

    w = np.zeros(L)
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(L, N):

        x_vec = x[n:n-L:-1]

        y[n] = np.dot(w, x_vec)

        e[n] = d[n] - y[n]

        w = w + mu * x_vec * e[n]

    return y, e

# --------------------------------------------------
# Plot MSE curves for different mu values
# --------------------------------------------------

window = 200

plt.figure(figsize=(12, 6))

for mu in mu_values:

    y, e = lms_filter(x, d, mu, L)

    # MSE between clean and cleaned signal
    error = s - e

    mse_smooth = np.convolve(
        error**2,
        np.ones(window) / window,
        mode='valid'
    )

    plt.plot(mse_smooth, label=f"mu = {mu:.4f}")

plt.xlabel("Sample n")

plt.text(
    0.98,
    0.75,
    f"$\\mu_{{max}}$ = {mu_max:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.title("MSE Curves for Different Mu Values")
plt.ylabel("Mean Squared Error")
plt.ylim(0, 1.5)
plt.grid()
plt.legend(
    loc='upper right',
    framealpha=0.9
)

plt.tight_layout()
plt.show()