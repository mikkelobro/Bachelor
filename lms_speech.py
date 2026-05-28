import numpy as np
import matplotlib.pyplot as plt
import librosa

# --------------------------------------------------
# Load clean audio signal
# --------------------------------------------------

clean_path = "Audio files/No noise/Mikkel_24år.wav"

s, fs = librosa.load(clean_path, sr=None, mono=True)

# Make sure signal is not too loud
s = s / np.max(np.abs(s))

# --------------------------------------------------
# Parameters
# --------------------------------------------------

N = len(s)
t = np.arange(N) / fs

L = 10
mu = 0.01

# --------------------------------------------------
# Non-stationary noise
# --------------------------------------------------

noise = (1 + 0.5*np.sin(2*np.pi*1*t)) * np.random.randn(N)

# Scale noise so it does not completely dominate speech
noise =  noise / np.max(np.abs(noise))  #normaliseret støj

# --------------------------------------------------
# Desired signal = clean speech + noise
# --------------------------------------------------

d = s + noise

# Reference noise
x = noise + 0.1*np.random.randn(N)

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

    x_vec = x[n:n-L:-1]

    y[n] = np.dot(w, x_vec)

    e[n] = d[n] - y[n]

    w = w + mu * x_vec * e[n]

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

print(f"SNR before LMS: {SNR_before:.2f} dB")
print(f"SNR after LMS: {SNR_after:.2f} dB")

# --------------------------------------------------
# Plot signals
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

axs[0].plot(s)
axs[0].set_title("Original Clean Speech Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(d)
axs[1].set_title("Noisy Speech Signal d(n)")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(e)
axs[3].set_title("Cleaned Speech Signal e(n)")
axs[3].set_xlabel("Sample n")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()