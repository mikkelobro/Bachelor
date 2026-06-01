import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

np.random.seed(1)

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

# --------------------------------------------------
# Choose one mu value for signal plots and SNR
# --------------------------------------------------

mu = 0.1 * mu_max

y, e = lms_filter(x, d, mu, L)

# --------------------------------------------------
# SNR before LMS
# --------------------------------------------------

noise_before = d - s

SNR_before = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_before**2)
)

# --------------------------------------------------
# SNR after LMS
# --------------------------------------------------

noise_after = e - s

SNR_after = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_after**2)
)

print(f"\nSelected mu for signal plots and SNR: {mu:.4f}")
print(f"SNR before LMS: {SNR_before:.2f} dB")
print(f"SNR after LMS: {SNR_after:.2f} dB")

# --------------------------------------------------
# Plot signals
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

axs[0].plot(s)
axs[0].set_title("Original Clean Chirp Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(d)
axs[1].set_title("Noisy Chirp Signal d(n)")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(e)
axs[3].set_title("Cleaned Chirp Signal e(n)")
axs[3].set_xlabel("Sample n")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()