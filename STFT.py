import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt

# Parameters
#test
fs = 1000
T = 2.0
t = np.linspace(0, T, int(fs*T), endpoint=False)

f0 = 10
f1 = 500

# Chirp signal
x = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# STFT parameters
N = 512
hop = N // 2
window = np.hamming(N)

# Number of frames
num_frames =  1 + (len(x) - N) // hop

# Frame buffer
frames = np.zeros((num_frames, N))

for i in range(num_frames):
    start = i * hop
    frames[i] = x[start:start + N] * window

# FFT
stft = np.fft.rfft(frames, axis=1)
magnitude = np.abs(stft)

# Axes
frequencies = np.fft.rfftfreq(N, d=1/fs)
times = (np.arange(num_frames) * hop) / fs

# Plot
plt.pcolormesh(times, frequencies, magnitude.T, shading='gouraud')
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude")
plt.show()