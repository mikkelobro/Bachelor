from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# --- Load original ---
fs1, x = wavfile.read("Mikkel_24år.wav")

if x.ndim > 1:
    x = np.mean(x, axis=1)

x = x / np.max(np.abs(x))

# --- Load denoised ---
fs2, x_denoised = wavfile.read("denoised.wav")

if x_denoised.ndim > 1:
    x_denoised = np.mean(x_denoised, axis=1)

x_denoised = x_denoised / np.max(np.abs(x_denoised))

# Ensure same sampling rate
assert fs1 == fs2
fs = fs1

# Time axis
time = np.arange(len(x)) / fs
sampling_period = np.diff(time).mean()

# --- perform CWT ---
wavelet = "cmor1.5-1.0"
widths = np.geomspace(1, 1024, num=100)

# Original
cwt_orig, freqs = pywt.cwt(x, widths, wavelet, sampling_period=sampling_period)
cwt_orig = np.abs(cwt_orig[:-1, :-1])

# Denoised
cwt_denoised, _ = pywt.cwt(x_denoised, widths, wavelet, sampling_period=sampling_period)
cwt_denoised = np.abs(cwt_denoised[:-1, :-1])

# --- plot result ---
fig, axs = plt.subplots(2, 1, figsize=(8,8))

# Original scalogram
pcm1 = axs[0].pcolormesh(time[:-1], freqs[:-1], cwt_orig)
axs[0].set_yscale("log")
axs[0].set_ylim(50, 4000)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Original Signal Scalogram")
fig.colorbar(pcm1, ax=axs[0])

# Denoised scalogram
pcm2 = axs[1].pcolormesh(time[:-1], freqs[:-1], cwt_denoised)
axs[1].set_yscale("log")
axs[1].set_ylim(50, 4000)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_title("Denoised Signal Scalogram")
fig.colorbar(pcm2, ax=axs[1])

plt.tight_layout()
plt.show()