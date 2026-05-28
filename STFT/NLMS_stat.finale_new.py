import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 10
beta = 0.01
c = 1e-6
max_duration = 10

# --------------------------------------------------
# Load clean signal s(n)
# --------------------------------------------------

s_file = "Audio files/No noise/Mikkel_24år.wav"
s, fs = librosa.load(s_file, sr=None, mono=True)

s = s[:int(max_duration * fs)]
s = s - np.mean(s)
s = s / np.max(np.abs(s))

# --------------------------------------------------
# Load desired signal d(n)
# --------------------------------------------------

d_file = "Audio files/With noise/noisy_nonstationary.wav"
d, fs = librosa.load(d_file, sr=None, mono=True)

d = d[:int(max_duration * fs)]
d = d - np.mean(d)
d = d / np.max(np.abs(d))

# --------------------------------------------------
# Load reference signal x_ref(n)
# --------------------------------------------------

x_ref_file = "Audio files/With noise/noise_nonstationary_only.wav"
x_ref, fs_ref = librosa.load(x_ref_file, sr=fs, mono=True)

x_ref = x_ref[:len(d)]
x_ref = x_ref - np.mean(x_ref)
x_ref = x_ref / np.max(np.abs(x_ref))

# Dampen noise reference
x_ref = 0.4 * x_ref

# --------------------------------------------------
# Make sure signals have same length
# --------------------------------------------------

N = min(len(d), len(x_ref))

d = d[:N]
x_ref = x_ref[:N]

n = np.arange(N)

# --------------------------------------------------
# Check correlation and alignment
# --------------------------------------------------

corr = np.corrcoef(d, x_ref)[0, 1]
print(f"Correlation between d and x_ref: {corr:.4f}")

# Plot first part of d and x_ref together
plt.figure(figsize=(12, 4))
plt.plot(n[:5000], d[:5000], label="d: speech-dominant")
plt.plot(n[:5000], x_ref[:5000], label="x_ref: noise-dominant", alpha=0.8)
plt.title("Comparison of d(n) and x_ref(n)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# NLMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Error signal / cleaned speech

# --------------------------------------------------
# NLMS algorithm
# --------------------------------------------------

for i in range(L, N):

    x_vec = x_ref[i:i-L:-1]

    y[i] = np.dot(w, x_vec)

    e[i] = d[i] - y[i]

    norm_factor = c + np.dot(x_vec, x_vec)

    w = w + (beta / norm_factor) * x_vec * e[i]
# --------------------------------------------------
# SNR calculation
# --------------------------------------------------

clean_file = "Audio files/No noise/Mikkel_24år.wav"
s_clean, fs_clean = librosa.load(clean_file, sr=fs, mono=True)

s_clean = s_clean[:N]
s_clean = s_clean - np.mean(s_clean)
s_clean = s_clean / np.max(np.abs(s_clean))

# Make sure all signals have same length
N_snr = min(len(s_clean), len(d), len(e))

s_clean_snr = s_clean[:N_snr]
d_snr = d[:N_snr]
e_snr = e[:N_snr]

snr_before = 10 * np.log10(
    np.sum(s_clean_snr**2) / np.sum((s_clean_snr - d_snr)**2)
)

snr_after = 10 * np.log10(
    np.sum(s_clean_snr**2) / np.sum((s_clean_snr - e_snr)**2)
)

snr_improvement = snr_after - snr_before

print(f"SNR before denoising: {snr_before:.2f} dB")
print(f"SNR after denoising: {snr_after:.2f} dB")
print(f"SNR improvement: {snr_improvement:.2f} dB")


# --------------------------------------------------
# Save as new file
# --------------------------------------------------

sf.write("Audio files/Denoised/NLMS_statinoary_finale.wav", e, fs) 
print("Filer gemt: NLMS_statinoary_finale.wav")

# --------------------------------------------------
# Plot
# --------------------------------------------------


fig, axs = plt.subplots(4, 1, figsize=(12, 12))

axs[0].plot(n, s)
axs[0].set_title("Clean Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(n, d)
axs[1].set_title("Desired Signal d(n): Noisy Speech")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(n, y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(n, e)
axs[3].set_title("Cleaned Speech / Error Signal e(n)")
axs[3].set_xlabel("Samples")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()