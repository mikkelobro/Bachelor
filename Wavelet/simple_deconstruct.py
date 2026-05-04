import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate signal

fs = 20000
t = np.arange(0, 1, 1/fs)

x = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*5000*t)

# Decomposition

wavelet = 'haar'
level = 6
coeffs = pywt.wavedec(x, wavelet, level=level)

# Zoom window (adjust as needed)
t_min = 0.02
t_max = 0.04
mask = (t >= t_min) & (t <= t_max)



# Detail spaces

D = {}

for j in range(1, level+1):
    coeffs_D = [np.zeros_like(c) for c in coeffs]
    
    idx = level - j + 1  # maps j → index
    coeffs_D[idx] = coeffs[idx]
    
    Dj = pywt.waverec(coeffs_D, wavelet)
    D[j] = Dj[:len(x)]

# Approximation spaces

V = {}

for j in range(0, level+1):

    coeffs_V = [coeffs[0]]  # always keep a_J

    for i in range(1, len(coeffs)):
        # keep details ABOVE level j
        if i <= level - j:
            coeffs_V.append(coeffs[i])
        else:
            coeffs_V.append(np.zeros_like(coeffs[i]))

    V[j] = pywt.waverec(coeffs_V, wavelet)[:len(x)]

# Plot side-by-side
fig, axs = plt.subplots(level+1, 2, figsize=(12, 16))

for j in range(0, level+1):

    # Approximation
    axs[j, 0].plot(t[mask], V[j][mask])
    axs[j, 0].set_ylabel(f"V{j}")

    # Detail
    if j > 0:
        axs[j, 1].plot(t[mask], D[j][mask])
        axs[j, 1].set_ylabel(f"D{j}")
    else:
        axs[j, 1].axis('off')

axs[0, 0].set_title("Approximation spaces")
axs[0, 1].set_title("Detail spaces")

for ax in axs[-1]:
    ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("wavelet/wavelet_decomposition.pdf")