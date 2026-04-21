import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
import numpy as np

fs1, x_orig = wavfile.read("Audio files/No noise/Mikkel_24år.wav")
fs2, x_noisy_file = wavfile.read("Audio files/With noise/noisy.wav")
fs3, x_sure_file = wavfile.read("Audio files/Denoised/sure_soft.wav")

def plot_scalogram(signal, title, fs):
    signal = signal.astype(float)
    signal = signal / np.max(np.abs(signal))

    # Time axis
    t = np.arange(len(signal)) / fs

    # Logarithmic scales
    scales = np.geomspace(1, 1024, num=100)

    # CWT
    cwtmatr, freqs = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/fs)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    # Plot
    pcm = plt.pcolormesh(t, freqs, cwtmatr)
    
    plt.yscale("log")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(pcm)


# Scalograms
plt.figure(figsize=(12, 10))

plt.subplot(3,1,1)
plot_scalogram(x_orig, "Scalogram - Original Signal", fs1)

plt.subplot(3,1,2)
plot_scalogram(x_noisy_file, "Scalogram - Noisy Signal", fs2)

plt.subplot(3,1,3)
plot_scalogram(x_sure_file, "Scalogram - Denoised (SURE + Soft)", fs3)

plt.tight_layout()
plt.show()