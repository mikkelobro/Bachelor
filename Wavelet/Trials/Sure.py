import numpy as np
from scipy.io import wavfile
import pywt

def sure_threshold(detail_coeffs, sigma):
    # Normalize coefficients by noise standard deviation
    normalized_coeffs = detail_coeffs / sigma

    # Squared normalized coefficient magnitudes
    coeffs_sq = np.sort(np.abs(normalized_coeffs) ** 2)

    # Number of coefficients
    n = len(coeffs_sq)

    # Cumulative sum
    cumulative_sum = np.cumsum(coeffs_sq)

    # Candidate SURE risks (normalized domain)
    sure_risks = np.zeros(n)

    for k in range(n):
        sure_risks[k] = (
            n
            - 2 * (k + 1)
            + cumulative_sum[k]
            + (n - k - 1) * coeffs_sq[k]
        )

    # Index of minimum risk
    min_index = np.argmin(sure_risks)

    # Convert threshold back to original scale
    threshold = sigma * np.sqrt(coeffs_sq[min_index])
    return threshold


def soft_threshold(detail_coeffs, threshold):
    return np.sign(detail_coeffs) * np.maximum(np.abs(detail_coeffs) - threshold, 0)

# Load WAV file
sample_rate, audio = wavfile.read("Audio files/With noise/noisy_stationary.wav")

# Wavelet settings
wavelet = "db4"
levels = 6

# Number of coarsest detail levels to exclude from thresholding
# 0 means threshold all detail levels
exclude_coarse_levels = 0

# Perform DWT decomposition
coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Split approximation and detail coefficients
approximation = coeffs[0]
details = coeffs[1:]

thresholded_details = []

# Detail levels excluded from thresholding
# details[0] corresponds to the coarsest detail level
excluded_details = details[:exclude_coarse_levels]

# Detail levels to threshold
processed_details = details[exclude_coarse_levels:]

# Apply SURE thresholding to each detail level
for detail in processed_details:

    # Estimate noise level for current detail level
    sigma = np.median(np.abs(detail)) / 0.6745

    # SURE threshold
    sure_thresh = sure_threshold(detail, sigma)

    # Universal threshold upper bound (classical SureShrink)
    universal_thresh = sigma * np.sqrt(2 * np.log(len(detail)))

    # Classical SureShrink threshold
    threshold = min(sure_thresh, universal_thresh)

    # Apply soft thresholding
    thresholded_detail = soft_threshold(detail, threshold)

    thresholded_details.append(thresholded_detail)

# Reconstruct coefficient list
thresholded_coeffs = [approximation] + excluded_details + thresholded_details

# Perform inverse DWT reconstruction
denoised_audio = pywt.waverec(thresholded_coeffs, wavelet=wavelet)

# Ensure reconstructed signal length matches original
denoised_audio = denoised_audio[:len(audio)]

# Convert back to 16-bit PCM
denoised_audio = np.int16(denoised_audio / np.max(np.abs(denoised_audio)) * 32767)

# Save denoised WAV file
wavfile.write("Audio files/Denoised/sure_soft.wav", sample_rate, denoised_audio)