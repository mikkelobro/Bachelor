from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Upload file
fs, x = wavfile.read("Mikkel_24år.wav")

# Convert to mono
if x.ndim > 1:
    x = np.mean(x, axis=1)

# Normalize
x = x / np.max(np.abs(x))



# Plot audio file
t = np.arange(len(x)) / fs

plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
#plt.show()

# CWT

# Time axis
time = np.arange(len(x)) / fs
sampling_period = 1 / fs
wavelet = "cmor1.5-1.0"

# Logarithmic scales (important)
widths = np.geomspace(1, 1024, num=100)

# Continuous Wavelet Transform
cwtmatr, freqs = pywt.cwt(x, widths, wavelet, sampling_period=sampling_period)

# Magnitude (scalogram)
S = np.log1p(np.abs(cwtmatr))   # log scaling for contrast

# Plot
plt.figure(figsize=(8,5))
plt.pcolormesh(time, freqs, S, shading='auto', cmap='inferno')

plt.yscale("log")
plt.ylim(50, 4000)   # speech frequency range

plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Scalogram (Morlet Wavelet)")
plt.colorbar(label="Log Amplitude")

plt.tight_layout()
plt.show()