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

# Apply DWT
level = 5   # typical choice for audio

coeffs = pywt.wavedec(x_noisy, 'Haar', level=level)

# Soft thresholding
coeffs_thresh = [coeffs[0]]

for d in coeffs[1:]:
    sigma_j = np.median(np.abs(d)) / 0.6745
    alpha = 0.8   # try 0.3 – 0.8
    lam_j = alpha * sigma_j * np.sqrt(2 * np.log(len(d)))

    d_thresh = np.sign(d) * np.maximum(np.abs(d) - lam_j, 0)
    coeffs_thresh.append(d_thresh)

# IDWT
x_denoised = pywt.waverec(coeffs_thresh, 'haar')

# Denoised wav file
x_out = np.int16(x_denoised / np.max(np.abs(x_denoised)) * 32767)
wavfile.write("denoised.wav", fs, x_out)