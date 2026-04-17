from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Upload file
fs, x = wavfile.read("Mikkel_24år.wav")

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
wavfile.write("noisy.wav", fs, x_out)

# Apply DWT
level = 9   # typical choice for audio

coeffs = pywt.wavedec(x_noisy, 'Haar', level=level)

# Thresholding
coeffs_thresh = [coeffs[0]]

def semi_soft(d, lam1, lam2):
    out = np.zeros_like(d)
    
    abs_d = np.abs(d)
    
    # Region 1: below lower threshold
    mask1 = abs_d <= lam1
    
    # Region 2: transition
    mask2 = (abs_d > lam1) & (abs_d < lam2)
    
    # Region 3: above upper threshold
    mask3 = abs_d >= lam2
    
    out[mask1] = 0
    out[mask2] = np.sign(d[mask2]) * (lam2 * (abs_d[mask2] - lam1) / (lam2 - lam1))
    out[mask3] = d[mask3]
    
    return out

for d in coeffs[1:]:
    sigma_j = np.median(np.abs(d)) / 0.6745
    alpha = 1   # Threshold adjust
    lam_j = alpha * sigma_j * np.sqrt(2 * np.log(len(d)))

    # Hard thresholding
    #d_thresh = d * (np.abs(d) > lam_j)

    # Soft thresholding
    #d_thresh = np.sign(d) * np.maximum(np.abs(d) - lam_j, 0)

    # Semi thresholding
    alpha = 0.5
    lam1 = alpha * sigma_j * np.sqrt(2 * np.log(len(d)))
    lam2 = 2 * lam1   # typical choice
    d_thresh = semi_soft(d, lam1, lam2)

    coeffs_thresh.append(d_thresh)

# IDWT
x_denoised = pywt.waverec(coeffs_thresh, 'haar')

# Denoised wav file
x_out = np.int16(x_denoised / np.max(np.abs(x_denoised)) * 32767)
wavfile.write("denoised.wav", fs, x_out)

snr_noisy = 10 * np.log10(
    np.sum(x**2) / np.sum((x - x_noisy)**2)
)
print("SNR (noisy):", snr_noisy, "dB")