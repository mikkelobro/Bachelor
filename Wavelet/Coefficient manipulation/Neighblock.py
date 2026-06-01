import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, test_signal):
    noise = clean_signal #- test_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

def neighblock(detail_coeffs, sigma, L0=8):
    n = len(detail_coeffs)
    output = np.zeros(n)

    lambda_param = 4.50524

    if L0 is None:
        L0 = max(2, int(np.floor(np.log(n) / 2)))

    L1 = max(1, L0 // 2)
    L = L0 + 2 * L1

    # Non-overlapping central blocks (jb)
    for start in range(0, n, L0):

        end = min(start + L0, n)

        # JB block
        big_start = max(0, start - L1)
        big_end = min(n, end + L1)

        big_block = detail_coeffs[big_start:big_end]

        # energy
        S2 = np.sum(big_block ** 2)

        if S2 <= 0:
            shrink = 0.0
        else:
            shrink = max(
                0.0,
                (S2 - lambda_param * L * sigma**2) / S2
            )

        # Shrinkage
        output[start:end] = shrink * detail_coeffs[start:end]

    return output

# Load file
sample_rate, audio = wavfile.read(
    "Audio files/With noise/Bil.wav"
)
audio = audio / np.max(np.abs(audio))

_, clean_audio = wavfile.read(
    "Audio files/No noise/Mikkel_24år.wav"
)
clean_audio = clean_audio / np.max(np.abs(clean_audio))

# DWT
wavelet = "db8"
levels = 6

coeffs = pywt.wavedec(audio, wavelet=wavelet, level=levels)

# Split coefficients
approximation = coeffs[0]
details = coeffs[1:]

thresholded_details = []

# NeighBlock thresholding
for i, detail in enumerate(details):

    # MAD
    sigma = np.median(np.abs(detail)) / 0.6745

    # Shrinkage
    thresholded_detail = neighblock(
        detail,
        sigma,
        L0=None
    )

    detail_level = levels - i

    print(
        f"Detail level D{detail_level}: "
        f"NeighBlock applied | "
        f"Sigma = {sigma:.6f}"
    )

    thresholded_details.append(thresholded_detail)

# Reconstruction
thresholded_coeffs = [approximation] + thresholded_details

denoised_audio = pywt.waverec(
    thresholded_coeffs,
    wavelet=wavelet
)

# Match signal lengths
denoised_audio = denoised_audio[:len(audio)]
clean_audio = clean_audio[:len(denoised_audio)]
audio = audio[:len(denoised_audio)]

plot_audio = denoised_audio.copy()

# SNR
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