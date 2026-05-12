import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
from scipy.signal import resample
import os

# ============================================================
# Generate signal
# ============================================================

fs = 2000
t = np.arange(0, 1, 1/fs)

x = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*400*t)

# Add noise
noise = np.random.randn(len(x))
x_noisy = x + noise

# ============================================================
# Wavelet settings
# ============================================================

wavelet = 'haar'
level = 6

# ============================================================
# NeighBlock parameters
# ============================================================

block_size = 8
energy_threshold = 0.05

# ============================================================
# Save audio helper
# ============================================================

def save_audio(signal, filename, fs_audio):

    signal = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal * 32767)

    wavfile.write(filename, fs_audio, signal_int16)

# ============================================================
# Output directory
# ============================================================

os.makedirs("wavelet", exist_ok=True)

# ============================================================
# Audio playback sampling rate
# ============================================================

fs_audio = 44100

# ============================================================
# Resample audio
# ============================================================

x_audio = resample(x, int(len(x) * fs_audio / fs))
x_noisy_audio = resample(x_noisy,
                         int(len(x_noisy) * fs_audio / fs))

save_audio(x_audio, "wavelet/original.wav", fs_audio)
save_audio(x_noisy_audio, "wavelet/noisy.wav", fs_audio)

# ============================================================
# Wavelet decomposition
# ============================================================

coeffs = pywt.wavedec(x_noisy, wavelet, level=level)

coeffs_mod = coeffs.copy()

# ============================================================
# NeighBlock denoising
# ============================================================

for level_idx in range(1, len(coeffs_mod)):

    detail = coeffs_mod[level_idx].copy()

    for start in range(0, len(detail), block_size):

        end = min(start + block_size, len(detail))

        block = detail[start:end]

        # Local block energy
        energy = np.sum(block**2)

        # Remove weak blocks
        if energy < energy_threshold:
            detail[start:end] = 0

    coeffs_mod[level_idx] = detail

# ============================================================
# Reconstruction
# ============================================================

x_denoised = pywt.waverec(coeffs_mod, wavelet)

x_denoised = x_denoised[:len(x)]

# ============================================================
# Save denoised audio
# ============================================================

x_denoised_audio = resample(
    x_denoised,
    int(len(x_denoised) * fs_audio / fs)
)

save_audio(
    x_denoised_audio,
    "wavelet/neighblock_denoised.wav",
    fs_audio
)

print("Saved: wavelet/neighblock_denoised.wav")

# ============================================================
# Plot comparison
# ============================================================

plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(t, x)
plt.title("Original Signal")

plt.subplot(3,1,2)
plt.plot(t, x_noisy)
plt.title("Noisy Signal")

plt.subplot(3,1,3)
plt.plot(t, x_denoised)
plt.title("NeighBlock Denoised Signal")

plt.tight_layout()
plt.show()