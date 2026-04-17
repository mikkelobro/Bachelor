from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# No noise signal
fs1, x = wavfile.read("Mikkel_24år.wav")
if x.ndim > 1:
    x = np.mean(x, axis=1)
x = x / np.max(np.abs(x))

# Denoised
fs2, x_denoised = wavfile.read("denoised.wav")
if x_denoised.ndim > 1:
    x_denoised = np.mean(x_denoised, axis=1)
x_denoised = x_denoised / np.max(np.abs(x_denoised))

# CWT
wavelet = "cmor1.5-1.0"
widths = np.geomspace(1, 1024, num=100)

# No noise
time1 = np.arange(len(x)) / fs1
sampling_period1 = np.diff(time1).mean()

cwt_orig, freqs1 = pywt.cwt(x, widths, wavelet, sampling_period=sampling_period1)
cwt_orig = np.log1p(np.abs(cwt_orig[:-1, :-1]))

# Denoised
time2 = np.arange(len(x_denoised)) / fs2
sampling_period2 = np.diff(time2).mean()

cwt_denoised, freqs2 = pywt.cwt(x_denoised, widths, wavelet, sampling_period=sampling_period2)
cwt_denoised = np.log1p(np.abs(cwt_denoised[:-1, :-1]))

# plot
fig, axs = plt.subplots(2, 1, figsize=(8,8))

# No noise
pcm1 = axs[0].pcolormesh(time1[:-1], freqs1[:-1], cwt_orig, cmap='inferno')
axs[0].set_yscale("log")
axs[0].set_ylim(50, 4000)
axs[0].set_title("Original Signal")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
fig.colorbar(pcm1, ax=axs[0])

# Denoised
pcm2 = axs[1].pcolormesh(time2[:-1], freqs2[:-1], cwt_denoised, cmap='inferno')
axs[1].set_yscale("log")
axs[1].set_ylim(50, 4000)
axs[1].set_title("Denoised Signal")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
fig.colorbar(pcm2, ax=axs[1])

plt.tight_layout()
plt.show()