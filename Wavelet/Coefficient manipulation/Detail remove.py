import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Load WAV file
sample_rate, audio = wavfile.read("Audio files/With noise/noisy_stationary.wav")
audio = audio / np.max(np.abs(audio))

# Load clean reference signal for SNR calculation
_, clean_audio = wavfile.read("Audio files/No noise/Mikkel_24år.wav")
clean_audio = clean_audio / np.max(np.abs(clean_audio))

# Wavelet settings
wavelet = "db8"
levels = 6

# Detail levels to remove completely
# Example: [1, 2] removes D1 and D2
remove_levels = [1,2,]

# Perform DWT decomposition
coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Approximation coefficients
approximation = coeffs[0]

# Detail coefficients
details = coeffs[1:]

processed_details = []

for i, detail in enumerate(details):

    # Convert index to actual detail level
    # details[0] = D6, details[1] = D5, ..., details[5] = D1
    detail_level = levels - i

    if detail_level in remove_levels:
        print(f"Removing detail level D{detail_level}")
        processed_detail = np.zeros_like(detail)
    else:
        print(f"Keeping detail level D{detail_level}")
        processed_detail = detail.copy()

    processed_details.append(processed_detail)

# Reconstruct coefficient list
modified_coeffs = [approximation] + processed_details

# Inverse DWT reconstruction
denoised_audio = pywt.waverec(modified_coeffs, wavelet=wavelet)

# Match original length
denoised_audio = denoised_audio[:len(audio)]
clean_audio = clean_audio[:len(denoised_audio)]
audio = audio[:len(denoised_audio)]

# Calculate SNR
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
plt.plot(time, denoised_audio, linewidth=1.0, label="Modified Signal")

plt.title("Wavelet Detail Level Removal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save result
wavfile.write(
    "Audio files/Denoised/detail_level_removal.wav",
    sample_rate,
    output_audio,
)