import numpy as np
import pywt
from scipy.io import wavfile

# --- Load audio ---
fs, x = wavfile.read("Audio files/No noise/Mikkel_24år.wav")

# Convert to mono
if x.ndim > 1:
    x = np.mean(x, axis=1)

# Normalize
x = x / np.max(np.abs(x))

# --- Add noise (optional) ---
noise_level = 0.05
noise = noise_level * np.random.randn(len(x))
x_noisy = x + noise

# --- Wavelet decomposition ---
wavelet = 'db4'
level = 9
coeffs = pywt.wavedec(x_noisy, wavelet, level=level)

# --- BayesShrink thresholding ---
coeffs_thresh = [coeffs[0]]  # keep approximation

# Estimate noise from finest scale
d1 = coeffs[-1]
sigma = np.median(np.abs(d1)) / 0.6745
sigma2 = sigma**2

for d in coeffs[1:]:
    # Total variance at level j
    var_d = np.mean(d**2)
    
    # Signal variance
    sigma_x = np.sqrt(max(var_d - sigma2, 0))
    
    # Compute BayesShrink threshold
    if sigma_x == 0:
        lam = np.max(np.abs(d))  # fallback
    else:
        lam = sigma2 / sigma_x
    
    # Apply soft thresholding
    d_thresh = pywt.threshold(d, lam, mode='soft')
    
    coeffs_thresh.append(d_thresh)

# --- Reconstruction ---
x_denoised = pywt.waverec(coeffs_thresh, wavelet)

# --- Normalize and save ---
x_out = np.int16(x_denoised / np.max(np.abs(x_denoised)) * 32767)
wavfile.write("Audio files/Denoised/denoised_bayesshrink.wav", fs, x_out)

# --- SNR calculation ---
snr_noisy = 10 * np.log10(np.sum(x**2) / np.sum((x - x_noisy)**2))
snr_denoised = 10 * np.log10(np.sum(x**2) / np.sum((x - x_denoised[:len(x)])**2))

print("SNR (noisy):", snr_noisy, "dB")
print("SNR (denoised):", snr_denoised, "dB")