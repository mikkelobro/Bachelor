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

s = s[:N]

n = np.arange(N)

# --------------------------------------------------
# Check correlation
# --------------------------------------------------

corr = np.corrcoef(d, x_ref)[0, 1]

print(f"Correlation between d and x_ref: {corr:.4f}")

# --------------------------------------------------
# Plot first part of signals
# --------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(n[:5000], d[:5000], label="d: speech-dominant")

plt.plot(
    n[:5000],
    x_ref[:5000],
    label="x_ref: noise-dominant",
    alpha=0.8
)

plt.title("Comparison of d(n) and x_ref(n)")

plt.xlabel("Samples")

plt.ylabel("Amplitude")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()

# --------------------------------------------------
# Upper bound for beta
# --------------------------------------------------

Px = np.mean(x_ref**2)

beta_max = 2 / (L * Px)

beta_conservative = 1 / (L * Px)

print("\n--------------------------------------------------")
print("Step-size stability estimate")
print("--------------------------------------------------")

print(f"Reference signal power Px = {Px:.6f}")

print(f"Theoretical upper bound beta_max = {beta_max:.4f}")

print(f"Conservative upper bound = {beta_conservative:.4f}")

print("--------------------------------------------------")

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

    # Filter output
    y[i] = np.dot(w, x_vec)

    # Error signal
    e[i] = d[i] - y[i]

    # Normalization factor
    norm_factor = c + np.dot(x_vec, x_vec)

    # Weight update
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

# --------------------------------------------------
# MSE curve
# --------------------------------------------------

window = 1000

mse = (s_clean_snr - e_snr)**2

mse_smooth = np.convolve(
    mse,
    np.ones(window) / window,
    mode='same'
)

plt.figure(figsize=(10,4))

plt.plot(mse_smooth)

plt.title("Smoothed MSE Curve for NLMS")

plt.xlabel("Samples")

plt.ylabel("Mean Squared Error")

plt.grid()

plt.tight_layout()

plt.show()

# --------------------------------------------------
# SNR before denoising
# --------------------------------------------------

snr_before = 10 * np.log10(
    np.sum(s_clean_snr**2)
    /
    np.sum((s_clean_snr - d_snr)**2)
)

# --------------------------------------------------
# SNR after denoising
# --------------------------------------------------

snr_after = 10 * np.log10(
    np.sum(s_clean_snr**2)
    /
    np.sum((s_clean_snr - e_snr)**2)
)

snr_improvement = snr_after - snr_before

print(f"\nSNR before denoising: {snr_before:.2f} dB")

print(f"SNR after denoising: {snr_after:.2f} dB")

print(f"SNR improvement: {snr_improvement:.2f} dB")

# --------------------------------------------------
# Save denoised signal
# --------------------------------------------------

sf.write(
    "Audio files/Denoised/NLMS_nonstationary_final.wav",
    e,
    fs
)

print("\nFile saved: NLMS_nonstationary_final.wav")

# --------------------------------------------------
# Plot signals
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# --------------------------------------------------
# Clean signal
# --------------------------------------------------

axs[0].plot(n, s)

axs[0].set_title("Clean Signal s(n)")

axs[0].set_ylabel("Amplitude")

axs[0].grid()

# --------------------------------------------------
# Noisy signal
# --------------------------------------------------

axs[1].plot(n, d)

axs[1].set_title(
    f"Desired Signal d(n): Noisy Speech | "
    f"SNR = {snr_before:.2f} dB"
)

axs[1].set_ylabel("Amplitude")

axs[1].grid()

# --------------------------------------------------
# Estimated noise
# --------------------------------------------------

axs[2].plot(n, y)

axs[2].set_title("Estimated Noise y(n)")

axs[2].set_ylabel("Amplitude")

axs[2].grid()

# --------------------------------------------------
# Cleaned signal
# --------------------------------------------------

axs[3].plot(n, e)

axs[3].set_title(
    f"Cleaned Speech / Error Signal e(n) | "
    f"SNR = {snr_after:.2f} dB"
)

axs[3].set_xlabel("Samples")

axs[3].set_ylabel("Amplitude")

axs[3].grid()

plt.tight_layout()

plt.show()