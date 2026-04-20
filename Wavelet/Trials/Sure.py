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

# --- Haar DWT ---
wavelet = 'haar'
level = 5
coeffs = pywt.wavedec(x_noisy, wavelet, level=level)

# --- Noise estimate (from finest scale) ---
d1 = coeffs[-1]
sigma = np.median(np.abs(d1)) / 0.6745
sigma2 = sigma**2

# --- SURE threshold function ---
def sure_threshold(d, sigma):
    d2 = d**2
    n = len(d)

    # Sort squared coefficients
    sorted_d2 = np.sort(d2)
    
    risks = []
    lambdas = np.sqrt(sorted_d2)

    for lam in lambdas:
        term1 = n * sigma2
        term2 = np.sum(np.minimum(d2, lam**2))
        term3 = 2 * sigma2 * np.sum(np.abs(d) <= lam)
        
        risk = term1 + term2 - term3
        risks.append(risk)

    # Choose lambda minimizing SURE
    lam_opt = lambdas[np.argmin(risks)]
    return lam_opt

# --- Apply SURE thresholding ---
coeffs_thresh = [coeffs[0]]  # keep approximation

for d in coeffs[1:]:
    lam = sure_threshold(d, sigma)
    
    # Soft thresholding (standard for SURE)
    d_thresh = pywt.threshold(d, lam, mode='soft')
    
    coeffs_thresh.append(d_thresh)

# --- Reconstruction ---
x_denoised = pywt.waverec(coeffs_thresh, wavelet)

# --- Save output ---
x_out = np.int16(x_denoised / np.max(np.abs(x_denoised)) * 32767)
wavfile.write("Audio files/Denoised/denoised_sure.wav", fs, x_out)

# --- SNR comparison ---
snr_noisy = 10 * np.log10(np.sum(x**2) / np.sum((x - x_noisy)**2))
snr_denoised = 10 * np.log10(np.sum(x**2) / np.sum((x - x_denoised[:len(x)])**2))

print("SNR (noisy):", snr_noisy, "dB")
print("SNR (denoised):", snr_denoised, "dB")