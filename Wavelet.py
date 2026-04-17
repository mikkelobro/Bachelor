from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#Upload file
fs, x = wavfile.read("Mikkel_24år.wav")

#Convert to mono
if x.ndim > 1:
    x = np.mean(x, axis=1)

#Normalize
x = x / np.max(np.abs(x))


t = np.arange(len(x)) / fs

plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()