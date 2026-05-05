from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Upload file
fs_og, x = wavfile.read("Audio files/No noise/Mikkel_24år.wav")
x = x.astype(float)
x = x / np.max(np.abs(x))

fs, x_noisy = wavfile.read("Audio files/With noise/noisy_stationary.wav")
x_noisy = x_noisy.astype(float)
x_noisy = x_noisy / np.max(np.abs(x_noisy))

print(fs_og)

# Apply DWT
wavelet = 'db2'
level = 6
d_remove = 4 # Detail space to be removed. used later in code
coeffs = pywt.wavedec(x_noisy, wavelet, level=level)

# Remove detail coefficients

# VisuShrink (compare thresholding functions)
coeffs_visu_hard = [coeffs[0]]
coeffs_visu_soft = [coeffs[0]]
coeffs_visu_semi = [coeffs[0]]

# Adaptive methods (compare threshold selection)
coeffs_sure_soft = [coeffs[0]]
coeffs_bayes_soft = [coeffs[0]]

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
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
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

# BayesShrink function
def bayes_threshold(d, sigma):
    d = np.asarray(d)
    
    # Total variance at level j
    var_d = np.mean(d**2)
    
    # Signal variance
    sigma_x = np.sqrt(max(var_d - sigma**2, 0))
    
    # Avoid division by zero
    if sigma_x < 1e-8:
        return np.max(np.abs(d))
    
    # BayesShrink threshold
    return sigma**2 / sigma_x

for d in coeffs[1:]:
    sigma_j = np.median(np.abs(d)) / 0.6745
    lam_j = sigma_j * np.sqrt(2 * np.log(len(d)))

    # Hard thresholding
    d_thresh_hard = d * (np.abs(d) > lam_j)
    coeffs_visu_hard.append(d_thresh_hard)

    # Soft thresholding
    d_thresh_soft = np.sign(d) * np.maximum(np.abs(d) - lam_j, 0)
    coeffs_visu_soft.append(d_thresh_soft)

    # Semi-soft
    lam1 = lam_j
    lam2 = 2 * lam1   
    d_thresh_semi = semi_soft(d, lam1, lam2)
    coeffs_visu_semi.append(d_thresh_semi)

    # Soft with SURE
    lam_sure = sure_threshold(d, sigma)
    d_thresh_sure = np.sign(d) * np.maximum(np.abs(d) - lam_sure, 0)
    coeffs_sure_soft.append(d_thresh_sure)

    # Soft with BayesShrink
    lam_bayes = bayes_threshold(d, sigma)
    d_thresh_bayes = np.sign(d) * np.maximum(np.abs(d) - lam_bayes, 0)
    coeffs_bayes_soft.append(d_thresh_bayes)

# Removing detail space
coeffs_mod = coeffs.copy()
coeffs_mod[-d_remove:] = [np.zeros_like(c) for c in coeffs[-d_remove:]]

# IDWT
x_visu_hard = pywt.waverec(coeffs_visu_hard, wavelet)
x_visu_soft = pywt.waverec(coeffs_visu_soft, wavelet)
x_visu_semi = pywt.waverec(coeffs_visu_semi, wavelet)

x_sure_soft = pywt.waverec(coeffs_sure_soft, wavelet)
x_bayes_soft = pywt.waverec(coeffs_bayes_soft, wavelet)

x_mod = pywt.waverec(coeffs_mod, wavelet)

# Save files
def save_audio(signal, filename):
    #signal = signal[:len(x)]  # ensure same length
    signal = signal / np.max(np.abs(signal))  # normalize
    x_out = np.int16(signal * 32767)
    wavfile.write(filename, fs, x_out)

save_audio(x_visu_hard, "Audio files/Denoised/visu_hard.wav")
save_audio(x_visu_soft, "Audio files/Denoised/visu_soft.wav")
save_audio(x_visu_semi, "Audio files/Denoised/visu_semi.wav")

save_audio(x_sure_soft, "Audio files/Denoised/sure_soft.wav")
save_audio(x_bayes_soft, "Audio files/Denoised/bayes_soft.wav")
save_audio(x_mod, "Audio files/Denoised/remove_detail.wav")


# SNR comparison
#def compute_snr(original, noisy):
#    return 10 * np.log10(
#        np.sum(original**2) / np.sum((original - noisy)**2)
#    )

#snr_noisy = compute_snr(x, x_noisy)

# VisuShrink
#snr_visu_hard = compute_snr(x, x_visu_hard[:len(x)])
#snr_visu_soft = compute_snr(x, x_visu_soft[:len(x)])
#snr_visu_semi = compute_snr(x, x_visu_semi[:len(x)])

# SURE and BayesShrink
#snr_sure = compute_snr(x, x_sure_soft[:len(x)])
#snr_bayes = compute_snr(x, x_bayes_soft[:len(x)])

#snr_mod = compute_snr(x, x_mod[:len(x)])

#print("SNR comparison (dB):")
#print(f"Noisy:        {snr_noisy:.2f}")

#print("\nVisuShrink:")
#print(f"Hard:         {snr_visu_hard:.2f}")
#print(f"Soft:         {snr_visu_soft:.2f}")
#print(f"Semi-soft:    {snr_visu_semi:.2f}")
#print(f"remove detail:    {snr_mod:.2f}")

#print("\nAdaptive (Soft threshold):")
#print(f"SURE:         {snr_sure:.2f}")
#print(f"BayesShrink:  {snr_bayes:.2f}")