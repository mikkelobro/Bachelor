import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

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
audio = audio / np.max(np.abs(audio))

# Load clean reference signal for SNR calculation
_, clean_audio = wavfile.read("Audio files/No noise/Mikkel_24år.wav")
clean_audio = clean_audio / np.max(np.abs(clean_audio))

# Wavelet settings
wavelet = "db10"
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
for i, detail in enumerate(processed_details):

    # Estimate noise level for current detail level
    sigma = np.median(np.abs(detail)) / 0.6745

    # SURE threshold
    sure_thresh = sure_threshold(detail, sigma)

    # Universal threshold upper bound (classical SureShrink)
    universal_thresh = sigma * np.sqrt(2 * np.log(len(audio)))

    # Classical SureShrink threshold
    threshold = min(sure_thresh, universal_thresh)

    # Determine which threshold was selected
    if sure_thresh <= universal_thresh:
        threshold_method = "SURE"
    else:
        threshold_method = "Universal"

    detail_level = levels - (i + exclude_coarse_levels)

    print(
        f"Detail level D{detail_level}: "
        f"{threshold_method} threshold selected | "
        f"Threshold = {threshold:.6f}"
    )


    # Apply soft thresholding
    thresholded_detail = soft_threshold(detail, threshold)

    thresholded_details.append(thresholded_detail)

# Reconstruct coefficient list
thresholded_coeffs = [approximation] + excluded_details + thresholded_details

# Perform inverse DWT reconstruction
denoised_audio = pywt.waverec(thresholded_coeffs, wavelet=wavelet)

# Ensure reconstructed signal length matches original and clean reference
denoised_audio = denoised_audio[:len(audio)]
clean_audio = clean_audio[:len(denoised_audio)]
audio = audio[:len(denoised_audio)]

# Store floating-point version for plotting
plot_audio = denoised_audio.copy()

# Calculate SNR before denoising
snr_before = calculate_snr(clean_audio, audio)

# Calculate SNR after denoising
snr_after = calculate_snr(clean_audio, denoised_audio)

print(f"SNR before denoising: {snr_before:.2f} dB")
print(f"SNR after denoising: {snr_after:.2f} dB")

# Convert back to 16-bit PCM
output_audio = np.int16(denoised_audio / np.max(np.abs(denoised_audio)) * 32767)

# Time axis
time = np.arange(len(audio)) / sample_rate

# Plot original and denoised signals on top of each other
plt.figure(figsize=(12, 5))

# Original noisy signal (faded)
plt.plot(time, audio, alpha=0.4, label="Original Noisy Signal")

# Denoised signal
plt.plot(time, plot_audio, linewidth=1.0, label="Denoised Signal")

plt.title("Comparison of Original and Denoised Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Save denoised WAV file
wavfile.write("Audio files/Denoised/sure_soft.wav", sample_rate, output_audio)