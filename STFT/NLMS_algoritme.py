import numpy as np
import matplotlib.pyplot as plt
import librosa

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 8
beta = 1.9          # NLMS can usually use a larger step size than LMS
c = 1e-6    # Avoid division by zero
max_duration = 10

# --------------------------------------------------
# Load clean audio signal
# --------------------------------------------------

# Time axis
t = np.arange(N)

# Clean sinus signal
s = np.sin(2*np.pi*0.01*t)

# --------------------------------------------------
# Create non-stationary noise
# --------------------------------------------------

noise = (1 + 0.5*np.sin(2*np.pi*1*t)) * np.random.randn(N)

noise = 0.05 * noise

d = s + noise

x_ref = noise + 0.01*np.random.randn(N)

# --------------------------------------------------
# NLMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Cleaned signal

# --------------------------------------------------
# NLMS algorithm
# --------------------------------------------------

for n in range(L, N):

    x_vec = x_ref[n:n-L:-1]

    y[n] = np.dot(w, x_vec)

    e[n] = d[n] - y[n]

    norm_factor = c + np.dot(x_vec, x_vec)

    w = w + (beta / norm_factor) * x_vec * e[n]

# --------------------------------------------------
# SNR calculation
# --------------------------------------------------

noise_before = d - s
SNR_before = 10*np.log10(np.mean(s**2) / np.mean(noise_before**2))

noise_after = e - s
SNR_after = 10*np.log10(np.mean(s**2) / np.mean(noise_after**2))

print(f"SNR before NLMS: {SNR_before:.2f} dB")
print(f"SNR after NLMS: {SNR_after:.2f} dB")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

axs[0].plot(t, s)
axs[0].set_title("Clean Audio Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(t, d)
axs[1].set_title(f"Noisy Audio Signal d(n) | SNR = {SNR_before:.2f} dB")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(t, y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(t, e)
axs[3].set_title(f"Cleaned Signal e(n) | SNR = {SNR_after:.2f} dB")
axs[3].set_xlabel("Time [s]")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()