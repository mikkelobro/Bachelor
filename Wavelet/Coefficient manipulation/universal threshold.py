import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Load File
sample_rate, audio = wavfile.read("Audio files/With noise/noisy_stationary.wav")
audio = audio / np.max(np.abs(audio))

_, clean_audio = wavfile.read("Audio files/No noise/Mikkel_24år.wav")
clean_audio = clean_audio / np.max(np.abs(clean_audio))

# DWT
wavelet = "db14"
levels = 6

coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Approximation coefficients
approximation = coeffs[0]

# Detail coefficients
details = coeffs[1:]

# MAD
sigma = np.median(np.abs(details[-1])) / 0.6745

processed_details = []

for i, detail in enumerate(details):

    detail_level = levels - i

    # Universal threshold
    N = len(detail)
    threshold = sigma * np.sqrt(2 * np.log(N))

    print(
        f"D{detail_level}: Threshold = {threshold:.6f}"
    )

    # Soft thresholding
    processed_detail = pywt.threshold(
        detail,
        value=threshold,
        mode="soft"
    )

    processed_details.append(processed_detail)

# IDWT
modified_coeffs = [approximation] + processed_details
denoised_audio = pywt.waverec(modified_coeffs, wavelet=wavelet)

# Match original length
denoised_audio = denoised_audio[:len(audio)]
clean_audio = clean_audio[:len(denoised_audio)]
audio = audio[:len(denoised_audio)]

# SNR
snr_before = calculate_snr(clean_audio, audio)
snr_after = calculate_snr(clean_audio, denoised_audio)

print(f"SNR before processing: {snr_before:.2f} dB")
print(f"SNR after processing: {snr_after:.2f} dB")

# Convert back to 16-bit PCM
output_audio = np.int16(
    denoised_audio / np.max(np.abs(denoised_audio)) * 32767
)

# Time axis
time = np.arange(len(audio)) / sample_rate

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(time, audio, alpha=0.4, label="Original Noisy Signal")
plt.plot(time, denoised_audio, linewidth=1.0, label="Denoised Signal")

plt.title("Wavelet Denoising Using Universal Threshold")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save result
wavfile.write(
    "Audio files/Denoised/universal_threshold.wav",
    sample_rate,
    output_audio,
)