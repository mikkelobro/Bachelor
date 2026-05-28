import numpy as np
from scipy.io import wavfile
import pywt

def sure_threshold(detail_coeffs, sigma):
    # Squared coefficient magnitudes
    coeffs_sq = np.sort(np.abs(detail_coeffs) ** 2)

    # Number of coefficients
    n = len(coeffs_sq)

    # Cumulative sum
    cumulative_sum = np.cumsum(coeffs_sq)

    # Candidate SURE risks
    sure_risks = np.zeros(n)

    for k in range(n):
        sure_risks[k] = (
            n * (sigma ** 2)
            + cumulative_sum[k]
            + (n - k - 1) * coeffs_sq[k]
            - 2 * (sigma ** 2) * (k + 1)
        )

    # Index of minimum risk
    min_index = np.argmin(sure_risks)

    # Optimal threshold
    threshold = np.sqrt(coeffs_sq[min_index])

    return threshold


def soft_threshold(detail_coeffs, threshold):
    return np.sign(detail_coeffs) * np.maximum(np.abs(detail_coeffs) - threshold, 0)

# Load WAV file
sample_rate, audio = wavfile.read("input.wav")

# Convert to float
audio = audio.astype(np.float32)

# Normalize
if np.max(np.abs(audio)) > 1:
    audio = audio / np.max(np.abs(audio))

# Convert stereo to mono
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Wavelet settings
wavelet = "db4"
levels = 6

# Perform DWT decomposition
coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Split approximation and detail coefficients
approximation = coeffs[0]
details = coeffs[1:]

# Estimate noise standard deviation from finest detail level
sigma = np.median(np.abs(details[-1])) / 0.6745

# Thresholded detail coefficients
thresholded_details = []

# Apply SURE thresholding to each detail level
for detail in details:
    threshold = sure_threshold(detail, sigma)
    thresholded_detail = soft_threshold(detail, threshold)
    thresholded_details.append(thresholded_detail)

# Reconstruct coefficient list
thresholded_coeffs = [approximation] + thresholded_details

# Perform inverse DWT reconstruction
denoised_audio = pywt.waverec(thresholded_coeffs, wavelet=wavelet)

# Ensure reconstructed signal length matches original
denoised_audio = denoised_audio[:len(audio)]

# Convert back to 16-bit PCM
denoised_audio = np.int16(denoised_audio / np.max(np.abs(denoised_audio)) * 32767)

# Save denoised WAV file
wavfile.write("denoised_output.wav", sample_rate, denoised_audio)