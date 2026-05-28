import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt


def calculate_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def neighblock(detail_coeffs, sigma, block_size=16, overlap=4):
    """
    NeighBlock thresholding for 1D wavelet detail coefficients.

    Parameters
    ----------
    detail_coeffs : ndarray
        Wavelet detail coefficients.
    sigma : float
        Estimated noise standard deviation.
    block_size : int
        Length of each neighborhood block.
    overlap : int
        Overlap between neighboring blocks.

    Returns
    -------
    ndarray
        Thresholded coefficients.
    """

    n = len(detail_coeffs)
    output = np.zeros(n)
    weight_sum = np.zeros(n)

    # Constant used in NeighBlock shrinkage
    lambda_param = 4.50524

    step = block_size - overlap

    for start in range(0, n, step):

        end = min(start + block_size, n)

        block = detail_coeffs[start:end]

        # Block energy
        block_energy = np.sum(block ** 2)

        # Shrinkage factor
        shrink = max(
            0,
            1 - (lambda_param * len(block) * sigma**2) / block_energy
        ) if block_energy > 0 else 0

        thresholded_block = shrink * block

        # Weighted overlap-add reconstruction
        output[start:end] += thresholded_block
        weight_sum[start:end] += 1

    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1

    return output / weight_sum


# Load noisy WAV file
sample_rate, audio = wavfile.read(
    "Audio files/With noise/noisy_nonstationary.wav"
)
audio = audio / np.max(np.abs(audio))

# Load clean reference
_, clean_audio = wavfile.read(
    "Audio files/No noise/Mikkel_24år.wav"
)
clean_audio = clean_audio / np.max(np.abs(clean_audio))

# Wavelet settings
wavelet = "db8"
levels = 6

# Perform wavelet decomposition
coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Split coefficients
approximation = coeffs[0]
details = coeffs[1:]

thresholded_details = []

# Apply NeighBlock thresholding
for i, detail in enumerate(details):

    # Noise estimate from MAD
    sigma = np.median(np.abs(detail)) / 0.6745

    # Apply NeighBlock shrinkage
    thresholded_detail = neighblock(
        detail,
        sigma,
        block_size=8,
        overlap= 4
    )

    detail_level = levels - i

    print(
        f"Detail level D{detail_level}: "
        f"NeighBlock applied | "
        f"Sigma = {sigma:.6f}"
    )

    thresholded_details.append(thresholded_detail)

# Reconstruct signal
thresholded_coeffs = [approximation] + thresholded_details

denoised_audio = pywt.waverec(
    thresholded_coeffs,
    wavelet=wavelet
)

# Match signal lengths
denoised_audio = denoised_audio[:len(audio)]
clean_audio = clean_audio[:len(denoised_audio)]
audio = audio[:len(denoised_audio)]

# Store for plotting
plot_audio = denoised_audio.copy()

# Compute SNR
snr_before = calculate_snr(clean_audio, audio)
snr_after = calculate_snr(clean_audio, denoised_audio)

print(f"SNR before denoising: {snr_before:.2f} dB")
print(f"SNR after denoising: {snr_after:.2f} dB")

# Convert to 16-bit PCM
output_audio = np.int16(
    denoised_audio / np.max(np.abs(denoised_audio)) * 32767
)

# Time axis
time = np.arange(len(audio)) / sample_rate

# Plot comparison
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    audio,
    alpha=0.4,
    label="Original Noisy Signal"
)

plt.plot(
    time,
    plot_audio,
    linewidth=1.0,
    label="Denoised Signal"
)

plt.title("Comparison of Original and NeighBlock Denoised Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save output
wavfile.write(
    "Audio files/Denoised/neighblock_soft.wav",
    sample_rate,
    output_audio
)