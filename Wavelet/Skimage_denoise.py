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
wavelet = 'db4'
levels = 6

# NeighBlock parameters
block_size = 16

# Choose which detail levels to remove completely
# d1 = finest detail coefficients
# d2 = second finest
# etc.
zero_detail_levels = []

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
# NeighBlock thresholding function
# Based on Cai & Silverman NeighBlock method
# -------------------------------------------------
def neighblock(detail, block_size, lambda_value=1.5053):

    n = len(detail)

    processed = np.zeros_like(detail)

    # Extension length
    extension = max(1, block_size // 2)

    # Process disjoint central blocks
    for start in range(0, n, block_size):

        end = min(start + block_size, n)

        # Central block b_i
        central_block = detail[start:end]

        # Extended neighbouring block B_i
        ext_start = max(0, start - extension)
        ext_end = min(n, end + extension)

        extended_block = detail[ext_start:ext_end]

        # Local energy estimate S^2
        S2 = np.sum(extended_block ** 2)

        # Number of coefficients in extended block
        L = len(extended_block)

        # Robust noise estimate using MAD
        sigma_local = np.median(np.abs(extended_block)) / 0.6745

        # NeighBlock shrinkage factor
        beta = max(
            0,
            1 - (lambda_value * L * sigma_local**2) / (S2 + 1e-10)
        )

        processed[start:end] = beta * central_block

    return processed

# -------------------------------------------------
# Denoising function
# -------------------------------------------------
def process_coefficients(method):

    new_coeffs = [coeffs[0]]

    for i, detail in enumerate(coeffs[1:], start=1):

        # Convert index to actual detail level
        detail_level = levels - i + 1

        # -----------------------------------------
        # Remove selected detail levels completely
        # -----------------------------------------
        if detail_level in zero_detail_levels:

            processed_detail = np.zeros_like(detail)

        else:

            # -------------------------------------
            # BayesShrink thresholding
            # -------------------------------------
            if method == 'bayes':

                var = np.var(detail)

                threshold = (sigma**2) / np.sqrt(
                    max(var - sigma**2, 1e-10)
                )

                processed_detail = pywt.threshold(
                    detail,
                    threshold,
                    mode='soft'
                )

            # -------------------------------------
            # NeighBlock thresholding
            # -------------------------------------
            elif method == 'neighblock':

                processed_detail = neighblock(
                    detail,
                    block_size=block_size
                )

            else:
                raise ValueError(
                    "method must be 'bayes' or 'neighblock'"
                )

        new_coeffs.append(processed_detail)

    reconstructed = pywt.waverec(new_coeffs, wavelet)

    return reconstructed[:len(signal)]

# -------------------------------------------------
# Generate both denoised signals
# -------------------------------------------------
denoised_bayes = process_coefficients('bayes')
denoised_neighblock = process_coefficients('neighblock')

# -------------------------------------------------
# Save denoised audio
# -------------------------------------------------
output_bayes = "Audio files/Denoised/bayeshrink_detail_remove.wav"
output_neighblock = "Audio files/Denoised/neighblock_detail_remove.wav"

sf.write(output_bayes, denoised_bayes, samplerate)
sf.write(output_neighblock, denoised_neighblock, samplerate)

print(f"BayesShrink audio saved as: {output_bayes}")
print(f"NeighBlock audio saved as: {output_neighblock}")

# -------------------------------------------------
# Plot BayesShrink result
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(signal, alpha=0.5, label="Noisy signal")
plt.plot(denoised_bayes, linewidth=1, label="BayesShrink")

plt.title("Wavelet Denoising (BayesShrink)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Plot NeighBlock result
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(signal, alpha=0.5, label="Noisy signal")
plt.plot(denoised_neighblock, linewidth=1, label="NeighBlock")

plt.title("Wavelet Denoising (NeighBlock)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()