import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt


def calculate_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def bayes_threshold(detail_coeffs, sigma):

    # Variance of noisy coefficients
    coeff_variance = np.mean(detail_coeffs ** 2)

    # Estimate signal variance
    signal_variance = max(coeff_variance - sigma**2, 0)

    # Signal standard deviation
    sigma_x = np.sqrt(signal_variance)

    # Avoid division by zero
    if sigma_x < 1e-10:
        return np.max(np.abs(detail_coeffs))

    # BayesShrink threshold
    threshold = sigma**2 / sigma_x

    return threshold


def soft_threshold(detail_coeffs, threshold):
    return np.sign(detail_coeffs) * np.maximum(
        np.abs(detail_coeffs) - threshold,
        0
    )


# Load WAV file
sample_rate, audio = wavfile.read(
    "Audio files/With noise/noisy_nonstationary.wav"
)

audio = audio / np.max(np.abs(audio))

# Load clean reference signal for SNR calculation
_, clean_audio = wavfile.read(
    "Audio files/No noise/Mikkel_24år.wav"
)

clean_audio = clean_audio / np.max(np.abs(clean_audio))

# Wavelet settings
wavelet = "db8"
levels = 6

# Number of coarsest detail levels to exclude from thresholding
exclude_coarse_levels = 0

# Perform DWT decomposition
coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Split approximation and detail coefficients
approximation = coeffs[0]
details = coeffs[1:]

thresholded_details = []

# Detail levels excluded from thresholding
excluded_details = details[:exclude_coarse_levels]

# Detail levels to threshold
processed_details = details[exclude_coarse_levels:]

# Apply BayesShrink thresholding to each detail level
for i, detail in enumerate(processed_details):

    # Estimate noise level using MAD estimator
    sigma = np.median(np.abs(detail)) / 0.6745

    # BayesShrink threshold
    threshold = bayes_threshold(detail, sigma)

    detail_level = levels - (i + exclude_coarse_levels)

    print(
        f"Detail level D{detail_level}: "
        f"BayesShrink threshold = {threshold:.6f}"
    )

    # Apply soft thresholding
    thresholded_detail = soft_threshold(detail, threshold)

    thresholded_details.append(thresholded_detail)

# Reconstruct coefficient list
thresholded_coeffs = (
    [approximation]
    + excluded_details
    + thresholded_details
)

# Perform inverse DWT reconstruction
denoised_audio = pywt.waverec(
    thresholded_coeffs,
    wavelet=wavelet
)

# Ensure reconstructed signal length matches original
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
output_audio = np.int16(
    denoised_audio / np.max(np.abs(denoised_audio)) * 32767
)

# Time axis
time = np.arange(len(audio)) / sample_rate

# Plot original and denoised signals
plt.figure(figsize=(12, 5))

# Original noisy signal
plt.plot(
    time,
    audio,
    alpha=0.4,
    label="Original Noisy Signal"
)

# Denoised signal
plt.plot(
    time,
    plot_audio,
    linewidth=1.0,
    label="Denoised Signal"
)

plt.title("Comparison of Original and BayesShrink Denoised Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

# Save denoised WAV file
wavfile.write(
    "Audio files/Denoised/bayes_soft.wav",
    sample_rate,
    output_audio
)