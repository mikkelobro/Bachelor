import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
import os
from scipy.signal import resample

# Generate signal

fs = 2048
t = np.arange(0, 1, 1/fs)

x = 1.0*np.sin(2*np.pi*64*t) + 0.5*np.sin(2*np.pi*128*t)

# Apply noise
noise_stat = 0.2 *np.random.randn(len(x))
x_noisy = x + noise_stat

# Decomposition

wavelet = 'Haar'
level = 5

# Remove finest detail space
d_remove = 1     # Removes D1 (finest detail space)

# ---------------------------
# Save audio helper
# ---------------------------
def save_audio(signal, filename):
    signal = signal / np.max(np.abs(signal))  # normalize
    signal_int16 = np.int16(signal * 32767)
    wavfile.write(filename, fs_audio, signal_int16)

os.makedirs("wavelet/Simple signal/Audio", exist_ok=True)

# ---------------------------
# Resample for audio playback
# ---------------------------
fs_audio = 44100

x_audio = resample(x, int(len(x) * fs_audio / fs))
x_noisy_audio = resample(x_noisy, int(len(x_noisy) * fs_audio / fs))

# Save original and noisy signals (audio)
save_audio(x_audio, "wavelet/Simple signal/Audio/original.wav")
save_audio(x_noisy_audio, "wavelet/Simple signal/Audio/noisy.wav")

# Zoom window (adjust as needed)
t_min = 0
t_max = 0.1
mask = (t >= t_min) & (t <= t_max)

# ---------------------------
# Function: decompose and plot
# ---------------------------
def decompose_and_plot(signal, coeffs_input, title_suffix, filename):
    # Detail spaces
    D_local = {}
    for j in range(1, level+1):
        coeffs_D = [np.zeros_like(c) for c in coeffs_input]
        idx = level - j + 1
        coeffs_D[idx] = coeffs_input[idx]
        Dj = pywt.waverec(coeffs_D, wavelet)
        D_local[j] = Dj[:len(signal)]

    # Approximation spaces
    V_local = {}
    for j in range(0, level+1):
        coeffs_V = [coeffs_input[0]]
        for i in range(1, len(coeffs_input)):
            if i <= level - j:
                coeffs_V.append(coeffs_input[i])
            else:
                coeffs_V.append(np.zeros_like(coeffs_input[i]))
        V_local[j] = pywt.waverec(coeffs_V, wavelet)[:len(signal)]

    # Global y-axis scaling for visual comparison
    y_lim = np.max(np.abs(signal)) * 1.1

    # Plot
    fig, axs = plt.subplots(level+1, 2, figsize=(14, 16))

    for j in range(0, level+1):

        # Approximation frequency band
        if j == 0:
            approx_label = f"A{j}\n[0-{fs/2:.0f} Hz]"
        else:
            approx_high = fs / (2**(j+1))
            approx_label = f"A{j}\n[0-{approx_high:.0f} Hz]"

        axs[j, 0].plot(t[mask], V_local[j][mask])
        axs[j, 0].set_ylabel(approx_label)
        axs[j, 0].set_ylim(-y_lim, y_lim)

        if j > 0:
            # Detail frequency band
            f_low = fs / (2**(j+1))
            f_high = fs / (2**j)

            axs[j, 1].plot(t[mask], D_local[j][mask])
            axs[j, 1].set_ylabel(f"D{j}\n[{f_low:.0f}-{f_high:.0f} Hz]")
            axs[j, 1].set_ylim(-y_lim, y_lim)
        else:
            axs[j, 1].axis('off')

    axs[0, 0].set_title(f"Approximation spaces {title_suffix}")
    axs[0, 1].set_title(f"Detail spaces {title_suffix}")

    for ax in axs[-1]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Original
coeffs = pywt.wavedec(x, wavelet, level=level)
decompose_and_plot(x, coeffs, "(original)", "wavelet/Simple signal/Plots/wavelet_decomposition.pdf")


# Make a copy of the coefficients
coeffs_mod = coeffs.copy()
# Set to zero the last coefficient of the copied list
coeffs_mod[-1] = 0*coeffs_mod[-1]

# Reconstruction
x_mod = pywt.waverec(coeffs_mod, wavelet)

x_mod_audio = resample(x_mod, int(len(x_mod) * fs_audio / fs))

print(f"Removing D1 to D{d_remove}")
save_audio(x_mod_audio, "wavelet/Simple signal/Audio/modified.wav")
decompose_and_plot(x, coeffs_mod, "(modified)", "wavelet/Simple signal/Plots/wavelet_decomposition_modified.pdf")

# Noisy
coeffs_noisy = pywt.wavedec(x_noisy, wavelet, level=level)
decompose_and_plot(x_noisy, coeffs_noisy, "(noisy)", "wavelet/Simple signal/Plots/wavelet_decomposition_noisy.pdf")