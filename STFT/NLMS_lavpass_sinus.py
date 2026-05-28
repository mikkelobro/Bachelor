import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --------------------------------------------------
# Parameters
# --------------------------------------------------

fs = 1000          # Sampling frequency
T = 2              # Signal duration [s]

N = int(fs * T)

L = 10
beta = 0.0005
c = 1e-6

# --------------------------------------------------
# Sample axis
# --------------------------------------------------

n = np.arange(N)

# --------------------------------------------------
# Time axis
# --------------------------------------------------

t = n / fs

# --------------------------------------------------
# Clean sinus signal
# --------------------------------------------------

f0 = 10

s = np.sin(2*np.pi*f0*t)

# --------------------------------------------------
# Create non-stationary noise
# --------------------------------------------------

noise = (1 + 0.5*np.sin(2*np.pi*1*t)) * np.random.randn(N)

# --------------------------------------------------
# Noisy signal
# --------------------------------------------------

d = s + noise

# --------------------------------------------------
# Reference noise through mild lowpass filter
# --------------------------------------------------

cutoff = 450  # Hz
order = 2

b, a = butter(order, cutoff / (fs / 2), btype="low")

x_ref = filtfilt(b, a, noise)

# --------------------------------------------------
# NLMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Cleaned signal

# --------------------------------------------------
# NLMS algorithm
# --------------------------------------------------

for i in range(L, N):

    x_vec = x_ref[i:i-L:-1]

    # Filter output
    y[i] = np.dot(w, x_vec)

    # Error signal
    e[i] = d[i] - y[i]

    # Normalization factor
    norm_factor = c + np.dot(x_vec, x_vec)

    # NLMS update
    w = w + (beta / norm_factor) * x_vec * e[i]

# --------------------------------------------------
# SNR calculation
# --------------------------------------------------

noise_before = d - s

SNR_before = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_before**2)
)

noise_after = e - s

SNR_after = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_after**2)
)

print(f"SNR before NLMS: {SNR_before:.2f} dB")
print(f"SNR after NLMS: {SNR_after:.2f} dB")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(5, 1, figsize=(12, 12))

# Clean signal
axs[0].plot(n, s)
axs[0].set_title("Clean Sinus Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Noisy signal
axs[1].plot(n, d)
axs[1].set_title("Noisy Signal d(n)")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

# Reference noise
axs[2].plot(n, x_ref)
axs[2].set_title("Filtered Reference Noise x_ref(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

# Estimated noise
axs[3].plot(n, y)
axs[3].set_title("Estimated Noise y(n)")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

# Cleaned signal
axs[4].plot(n, e)
axs[4].set_title("Cleaned Signal e(n)")
axs[4].set_xlabel("Samples")
axs[4].set_ylabel("Amplitude")
axs[4].grid()

plt.tight_layout()
plt.show()