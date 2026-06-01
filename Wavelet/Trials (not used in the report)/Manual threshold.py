import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
import os

# ---------------------------
# SNR helper
# ---------------------------
def compute_snr(reference, estimate):
    signal_power = np.sum(reference**2)
    noise_power = np.sum((reference - estimate)**2)
    return 10 * np.log10(signal_power / noise_power)

# ---------------------------
# Load WAV file
# ---------------------------
audio_path = "Audio files/With noise/noisy_bandpass_nonstationary.wav"

fs, x = wavfile.read(audio_path)

# Convert to float
x = x.astype(np.float64)

# If stereo -> convert to mono
if len(x.shape) > 1:
    x = np.mean(x, axis=1)

# Normalize
x = x / np.max(np.abs(x))

# Time axis
t = np.arange(len(x)) / fs

# ---------------------------
# Add Gaussian noise
# ---------------------------
noise_level = 0.02

noise = noise_level * np.random.randn(len(x))
x_noisy = x + noise

# ---------------------------
# Wavelet parameters
# ---------------------------
wavelet = 'db6'
level = 5

# ---------------------------
# Save audio helper
# ---------------------------
def save_audio(signal, filename):
    signal = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal * 32767)
    wavfile.write(filename, fs, signal_int16)

# Create folders
os.makedirs("Wavelet/Wav file/Audio", exist_ok=True)
os.makedirs("Wavelet/Wav file/Plots", exist_ok=True)

# Save original and noisy signals
save_audio(x, "Wavelet/Wav file/Audio/original.wav")
save_audio(x_noisy, "Wavelet/Wav file/Audio/noisy.wav")

# ---------------------------
# Zoom window for plots
# ---------------------------
t_min = 0
t_max = 0.05
mask = (t >= t_min) & (t <= t_max)

# ---------------------------
# Decompose and plot
# ---------------------------
def decompose_and_plot(signal, coeffs_input, title_suffix, filename):

    # Detail spaces
    D_local = {}

    for j in range(1, level + 1):

        coeffs_D = [np.zeros_like(c) for c in coeffs_input]

        idx = level - j + 1

        coeffs_D[idx] = coeffs_input[idx]

        Dj = pywt.waverec(coeffs_D, wavelet)

        D_local[j] = Dj[:len(signal)]

    # Approximation spaces
    V_local = {}

    for j in range(0, level + 1):

        coeffs_V = [coeffs_input[0]]

        for i in range(1, len(coeffs_input)):

            if i <= level - j:
                coeffs_V.append(coeffs_input[i])

            else:
                coeffs_V.append(np.zeros_like(coeffs_input[i]))

        V_local[j] = pywt.waverec(coeffs_V, wavelet)[:len(signal)]

    # Normalize all subplots individually
    def normalize(sig):
        max_val = np.max(np.abs(sig))

        if max_val == 0:
            return sig

        return sig / max_val

    # Plot
    fig, axs = plt.subplots(level + 1, 2, figsize=(14, 16))

    for j in range(0, level + 1):

        # Approximation labels
        if j == 0:
            approx_label = f"A{j}\n[0-{fs/2:.0f} Hz]"

        else:
            approx_high = fs / (2**(j + 1))
            approx_label = f"A{j}\n[0-{approx_high:.0f} Hz]"

        axs[j, 0].plot(t[mask], normalize(V_local[j])[mask])

        axs[j, 0].set_ylabel(approx_label)

        # Detail labels
        if j > 0:

            f_low = fs / (2**(j + 1))
            f_high = fs / (2**j)

            axs[j, 1].plot(t[mask], normalize(D_local[j])[mask])

            axs[j, 1].set_ylabel(
                f"D{j}\n[{f_low:.0f}-{f_high:.0f} Hz]"
            )

        else:
            axs[j, 1].axis('off')

    axs[0, 0].set_title(
        f"Approximation spaces {title_suffix}"
    )

    axs[0, 1].set_title(
        f"Detail spaces {title_suffix}"
    )

    for ax in axs[-1]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close(fig)

# ---------------------------
# Original decomposition
# ---------------------------
coeffs = pywt.wavedec(x, wavelet, level=level)

decompose_and_plot(
    x,
    coeffs,
    "(original)",
    "Wavelet/Wav file/Plots/decomposition_original.pdf"
)

# ---------------------------
# Noisy decomposition
# ---------------------------
coeffs_noisy = pywt.wavedec(x_noisy, wavelet, level=level)

decompose_and_plot(
    x_noisy,
    coeffs_noisy,
    "(noisy)",
    "Wavelet/Wav file/Plots/decomposition_noisy.pdf"
)

# ---------------------------
# Remove D1
# ---------------------------
coeffs_mod = coeffs_noisy.copy()

coeffs_mod[-1] = 0 * coeffs_mod[-1]

# Reconstruction
x_denoised = pywt.waverec(coeffs_mod, wavelet)

x_denoised = x_denoised[:len(x)]

# Save denoised audio
save_audio(
    x_denoised,
    "Wavelet/Wav file/Audio/denoised.wav"
)

# ---------------------------
# SNR
# ---------------------------
snr_noisy = compute_snr(x, x_noisy)

snr_denoised = compute_snr(x, x_denoised)

print(f"SNR noisy: {snr_noisy:.2f} dB")

print(f"SNR denoised: {snr_denoised:.2f} dB")

# ---------------------------
# Denoised decomposition
# ---------------------------
decompose_and_plot(
    x_denoised,
    coeffs_mod,
    "(denoised)",
    "Wavelet/Wav file/Plots/decomposition_denoised.pdf"
)