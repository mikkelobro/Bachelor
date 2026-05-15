import numpy as np
import soundfile as sf
import pywt
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma

# -------------------------------------------------
# Load WAV file
# -------------------------------------------------
input_file = "Audio files/With noise/noisy_nonstationary.wav"

signal, samplerate = sf.read(input_file)

# -------------------------------------------------
# Convert stereo to mono if necessary
# -------------------------------------------------
if signal.ndim > 1:
    signal = np.mean(signal, axis=1)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
wavelet = 'db10'
levels = 6

# Choose which detail levels to remove completely
# d1 = finest detail coefficients
# d2 = second finest
# etc.
zero_detail_levels = [1, 2]

# -------------------------------------------------
# Wavelet decomposition
# -------------------------------------------------
coeffs = pywt.wavedec(signal, wavelet, level=levels)

# coeffs structure:
# coeffs[0]  -> approximation coefficients
# coeffs[1]  -> d6
# coeffs[2]  -> d5
# ...
# coeffs[-1] -> d1

# -------------------------------------------------
# Estimate noise level
# -------------------------------------------------
sigma = estimate_sigma(signal)

# -------------------------------------------------
# Process coefficients
# -------------------------------------------------
new_coeffs = [coeffs[0]]

for i, detail in enumerate(coeffs[1:], start=1):

    # Convert index to actual detail level
    detail_level = levels - i + 1

    # ---------------------------------------------
    # Remove selected detail levels completely
    # ---------------------------------------------
    if detail_level in zero_detail_levels:

        processed_detail = np.zeros_like(detail)

    else:
        # -----------------------------------------
        # BayesShrink threshold
        # -----------------------------------------
        var = np.var(detail)

        threshold = (sigma**2) / np.sqrt(
            max(var - sigma**2, 1e-10)
        )

        # Soft thresholding
        processed_detail = pywt.threshold(
            detail,
            threshold,
            mode='soft'
        )

    new_coeffs.append(processed_detail)

# -------------------------------------------------
# Reconstruct signal
# -------------------------------------------------
denoised_signal = pywt.waverec(new_coeffs, wavelet)

# Match original length
denoised_signal = denoised_signal[:len(signal)]

# -------------------------------------------------
# Save denoised audio
# -------------------------------------------------
output_file = "Audio files/Denoised/bayeshrink_detail_remove.wav"

sf.write(output_file, denoised_signal, samplerate)

print(f"Denoised audio saved as: {output_file}")

# -------------------------------------------------
# Plot comparison
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(signal, alpha=0.5, label="Noisy signal")
plt.plot(denoised_signal, linewidth=1, label="Denoised signal")

plt.title("Wavelet Denoising with Selective Detail Removal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()