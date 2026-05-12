import numpy as np
import soundfile as sf
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load WAV file
# -------------------------------------------------
# Replace with your own file name
input_file = "Audio files/With noise/noisy_stationary.wav"

# Read audio
signal, samplerate = sf.read(input_file)

# -------------------------------------------------
# Convert stereo to mono if necessary
# -------------------------------------------------
if signal.ndim > 1:
    signal = np.mean(signal, axis=1)

# -------------------------------------------------
# Wavelet denoising
# -------------------------------------------------
denoised_signal = denoise_wavelet(
    signal,
    method='BayesShrink',   # Thresholding method
    mode='soft',            # Soft thresholding
    wavelet='db4',          # Daubechies 2 wavelet
    wavelet_levels=6,       # Number of decomposition levels
    rescale_sigma=True
)

# -------------------------------------------------
# Save denoised audio
# -------------------------------------------------
output_file = "Audio files/Denoised/skimage_denoised_stationary.wav"

sf.write(output_file, denoised_signal, samplerate)

print(f"Denoised audio saved as: {output_file}")

# -------------------------------------------------
# Plot comparison
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(signal, alpha=0.5, label="Noisy signal")
plt.plot(denoised_signal, linewidth=1, label="Denoised signal")

plt.title("Wavelet Denoising of Audio")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()