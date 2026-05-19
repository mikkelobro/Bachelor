import numpy as np
import matplotlib.pyplot as plt
import librosa

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 8
mu = 0.01
max_duration = 10

# --------------------------------------------------
# Load clean audio signal
# --------------------------------------------------

file_path = "Audio files/No noise/Mikkel_24år.wav"
s, fs = librosa.load(file_path, sr=None, mono=True)

s = s[:int(max_duration * fs)]
s = s - np.mean(s)

N = len(s)
t = np.arange(N) / fs

# --------------------------------------------------
# Create non-stationary noise
# --------------------------------------------------

noise = (1 + 0.5*np.sin(2*np.pi*1*t)) * np.random.randn(N)

# Scale noise down a bit
noise = 0.05 * noise

# Noisy signal
d = s + noise

# Reference noise
x_ref = noise + 0.01*np.random.randn(N)

# --------------------------------------------------
# LMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Cleaned signal

# --------------------------------------------------
# LMS algorithm
# --------------------------------------------------

for n in range(L, N):

    x_vec = x_ref[n:n-L:-1]

    y[n] = np.dot(w, x_vec)

    e[n] = d[n] - y[n]

    w = w + mu * x_vec * e[n]

# --------------------------------------------------
# SNR calculation
# --------------------------------------------------

noise_before = d - s
SNR_before = 10*np.log10(np.mean(s**2) / np.mean(noise_before**2))

noise_after = e - s
SNR_after = 10*np.log10(np.mean(s**2) / np.mean(noise_after**2))

print(f"SNR before LMS: {SNR_before:.2f} dB")
print(f"SNR after LMS: {SNR_after:.2f} dB")

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