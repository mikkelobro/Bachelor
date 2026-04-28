import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile

# Load signal
fs, x = wavfile.read("Audio files/With noise/noisy_stationary.wav")
x = x.astype(float)

# DWT
wavelet = 'db2'
level = 5
coeffs = pywt.wavedec(x, wavelet, level=level)

fig, axs = plt.subplots(level, 2, figsize=(10, 12))

start = 0.72
duration = 0.02
end = start + duration
t = np.arange(len(x)) / fs

for j in range(1, level+1):

    # Approximation
    coeffs_A = [coeffs[0]]

    for i in range(1, level+1):
        if i <= level - j:
            coeffs_A.append(coeffs[i])
        else:
            coeffs_A.append(np.zeros_like(coeffs[i]))

    A = pywt.waverec(coeffs_A, wavelet)
    A = A[:len(x)]

    # Detail
    coeffs_D = [np.zeros_like(c) for c in coeffs]
    coeffs_D[level - j + 1] = coeffs[level - j + 1]
    
    D = pywt.waverec(coeffs_D, wavelet)
    D = D[:len(x)]

    # Plot
    axs[j-1, 0].step(t, A, where='post')
    axs[j-1, 0].set_ylabel(f"A_{j}")
    axs[j-1, 0].set_xlim(start, end)

    axs[j-1, 1].step(t, D, where='post')
    axs[j-1, 1].set_ylabel(f"D_{j}")
    axs[j-1, 1].set_xlim(start, end)

# Labels
axs[0,0].set_title("Approximations")
axs[0,1].set_title("Details")

for ax in axs[-1]:
    ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()
