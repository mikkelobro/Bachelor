from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Upload file
fs, x = wavfile.read("Audio files/No noise/Mikkel_24år.wav")

# Convert to mono
if x.ndim > 1:
    x = np.mean(x, axis=1)

# Normalize
x = x / np.max(np.abs(x))

# Aply noise
noise_level = 0.05
noise = noise_level * np.random.randn(len(x))
x_noisy = x + noise

x_out = np.int16(x_noisy / np.max(np.abs(x_noisy)) * 32767)
wavfile.write("Audio files/With noise/noisy.wav", fs, x_out)

# Apply DWT
wavelet = 'Haar'
level = 4
coeffs = pywt.wavedec(x_noisy, wavelet, level=level)

# Thresholding
coeffs_thresh_hard = [coeffs[0]] # Keeps approximation unchanged
coeffs_thresh_soft = [coeffs[0]] 
coeffs_thresh_semi = [coeffs[0]] 
coeffs_thresh_sure = [coeffs[0]]

# Semi soft function
def semi_soft(d, lam1, lam2):
    out = np.zeros_like(d)
    
    for i in range(len(d)):
        val = d[i]
        abs_val = abs(val)
        
        if abs_val <= lam1:
            out[i] = 0
        
        elif abs_val < lam2:
            out[i] = np.sign(val) * (lam2 * (abs_val - lam1) / (lam2 - lam1))
        
        else:
            out[i] = val
    
    return out

# SURE function
def sure_threshold(d, sigma):
    d = np.asarray(d)
    n = len(d)
    d2 = d**2

    d2_sorted = np.sort(d2)
    lambdas = np.sqrt(d2_sorted)

    cumsum_d2 = np.cumsum(d2_sorted)
    k = np.arange(1, n+1)

    risk = (
        n * sigma**2
        + cumsum_d2
        + (n - k) * lambdas**2
        - 2 * sigma**2 * k
    )

    return lambdas[np.argmin(risk)]

for d in coeffs[1:]:
    sigma_j = np.median(np.abs(d)) / 0.6745
    lam_j = sigma_j * np.sqrt(2 * np.log(len(d)))

    # Hard thresholding
    d_thresh_hard = d * (np.abs(d) > lam_j)
    coeffs_thresh_hard.append(d_thresh_hard)

    # Soft thresholding
    d_thresh_soft = np.sign(d) * np.maximum(np.abs(d) - lam_j, 0)
    coeffs_thresh_soft.append(d_thresh_soft)

    # Semi-soft
    lam1 = lam_j
    lam2 = 2 * lam1   
    d_thresh_semi = semi_soft(d, lam1, lam2)
    coeffs_thresh_semi.append(d_thresh_semi)

    # SURE thresholding
    lam_sure = sure_threshold(d, sigma_j)
    d_thresh_sure = np.sign(d) * np.maximum(np.abs(d) - lam_sure, 0)
    coeffs_thresh_sure.append(d_thresh_sure)

# IDWT
x_denoised_hard = pywt.waverec(coeffs_thresh_hard, wavelet)
x_denoised_soft = pywt.waverec(coeffs_thresh_soft, wavelet)
x_denoised_semi = pywt.waverec(coeffs_thresh_semi, wavelet)
x_denoised_sure = pywt.waverec(coeffs_thresh_sure, wavelet)

# Save files
def save_audio(signal, filename):
    signal = signal[:len(x)]  # ensure same length
    signal = signal / np.max(np.abs(signal))  # normalize
    x_out = np.int16(signal * 32767)
    wavfile.write(filename, fs, x_out)

save_audio(x_denoised_hard, "Audio files/Denoised/hard.wav")
save_audio(x_denoised_soft, "Audio files/Denoised/soft.wav")
save_audio(x_denoised_semi, "Audio files/Denoised/semi.wav")
save_audio(x_denoised_sure, "Audio files/Denoised/sure.wav")

# SNR comparison
def compute_snr(original, noisy):
    return 10 * np.log10(
        np.sum(original**2) / np.sum((original - noisy)**2)
    )

snr_noisy = compute_snr(x, x_noisy)

snr_hard = compute_snr(x, x_denoised_hard[:len(x)])
snr_soft = compute_snr(x, x_denoised_soft[:len(x)])
snr_semi = compute_snr(x, x_denoised_semi[:len(x)])
snr_sure = compute_snr(x, x_denoised_sure[:len(x)])

print("SNR comparison (dB):")
print(f"Noisy: {snr_noisy:.2f}")
print(f"Hard:  {snr_hard:.2f}")
print(f"Soft:  {snr_soft:.2f}")
print(f"Semi:  {snr_semi:.2f}")
print(f"SURE:  {snr_sure:.2f}")