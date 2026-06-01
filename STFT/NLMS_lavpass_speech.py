import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 10
beta = 1.5
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
d, fs_d = librosa.load(d_file, sr=fs, mono=True)

d = d[:int(max_duration * fs)]
d = d - np.mean(d)
d = d / np.max(np.abs(d))

# --------------------------------------------------
# Load reference signal x_ref(n)
# --------------------------------------------------

x_ref_file = "Audio files/With noise/noise_nonstationary_only.wav"
x_ref, fs_ref = librosa.load(x_ref_file, sr=fs, mono=True)

x_ref = x_ref[:int(max_duration * fs)]
x_ref = x_ref - np.mean(x_ref)
x_ref = x_ref / np.max(np.abs(x_ref))

# --------------------------------------------------
# Lowpass filter reference noise
# --------------------------------------------------

cutoff = 20000  # Hz
order = 2

nyquist = fs / 2

print(f"Sampling frequency: {fs} Hz")
print(f"Nyquist frequency: {nyquist} Hz")
print(f"Chosen cutoff frequency: {cutoff} Hz")
# ------------------------------------------------

b, a = butter(order, cutoff / (fs / 2), btype="low")

x_ref = filtfilt(b, a, x_ref)

# --------------------------------------------------
# Make sure all signals have same length
# --------------------------------------------------

N = min(len(s), len(d), len(x_ref))

s = s[:N]
d = d[:N]
x_ref = x_ref[:N]

n = np.arange(N)

# --------------------------------------------------
# Correlation check
# --------------------------------------------------

corr = np.corrcoef(d, x_ref)[0, 1]
print(f"Correlation between d and x_ref: {corr:.4f}")

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

# Noise before denoising
noise_before = d - s

# Noise after denoising
noise_after = e - s

# Signal power
signal_power = np.mean(s**2)

# Noise power
noise_power_before = np.mean(noise_before**2)
noise_power_after = np.mean(noise_after**2)

# SNR values
snr_before = 10 * np.log10(signal_power / noise_power_before)
snr_after = 10 * np.log10(signal_power / noise_power_after)

# Improvement
snr_improvement = snr_after - snr_before

print("\n------------------------------")
print("NLMS DENOISING RESULTS")
print("------------------------------")
print(f"SNR before denoising : {snr_before:.2f} dB")
print(f"SNR after denoising  : {snr_after:.2f} dB")
print(f"Improvement          : {snr_improvement:.2f} dB")
print("------------------------------")

# --------------------------------------------------
# MSE curve
# --------------------------------------------------

window = 1000

mse_smooth = np.convolve(
    e**2,
    np.ones(window) / window,
    mode='same'
)

plt.figure(figsize=(10,4))

plt.plot(mse_smooth)

plt.title(
    f"Smoothed MSE Curve for NLMS\n"
    f"SNR Before = {snr_before:.2f} dB, "
    f"SNR After = {snr_after:.2f} dB, "
    f"Improvement = {snr_improvement:.2f} dB"
)

plt.xlabel("Samples")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Save denoised file
# --------------------------------------------------

sf.write("Audio files/Denoised/NLMS_nonstationary_audio.wav", e, fs)

print("File saved: NLMS_nonstationary_audio.wav")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Clean signal
axs[0].plot(n, s)
axs[0].set_title("Clean Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Desired signal
axs[1].plot(n, d)
axs[1].set_title("Desired Signal d(n): Noisy Speech")
axs[1].set_ylabel("Amplitude")
axs[1].grid()


# Estimated noise
axs[2].plot(n, y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

# Cleaned signal
axs[3].plot(n, e)
axs[3].set_title("Cleaned Signal e(n)")
axs[3].set_xlabel("Samples")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()

plt.figtext(
    0.15,
    0.01,
    f"SNR Before: {snr_before:.2f} dB    "
    f"SNR After: {snr_after:.2f} dB    "
    f"Improvement: {snr_improvement:.2f} dB",
    fontsize=10
)