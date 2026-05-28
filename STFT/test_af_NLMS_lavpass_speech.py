import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import os

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 10
beta = 0.1
c = 1e-6
max_duration = 10

cutoff = 20000   # Hz
order = 2
damp_factor = 0.4

# --------------------------------------------------
# Load signals
# --------------------------------------------------

s_file = "Audio files/No noise/Mikkel_24år.wav"
d_file = "Audio files/With noise/noisy_nonstationary.wav"
x_ref_file = "Audio files/With noise/noise_nonstationary_only.wav"

s, fs = librosa.load(s_file, sr=None, mono=True)
d, fs_d = librosa.load(d_file, sr=fs, mono=True)
x_ref_original, fs_ref = librosa.load(x_ref_file, sr=fs, mono=True)

# Cut duration
max_samples = int(max_duration * fs)

s = s[:max_samples]
d = d[:max_samples]
x_ref_original = x_ref_original[:max_samples]

# Same length
N = min(len(s), len(d), len(x_ref_original))

s = s[:N]
d = d[:N]
x_ref_original = x_ref_original[:N]

# Remove DC
s = s - np.mean(s)
d = d - np.mean(d)
x_ref_original = x_ref_original - np.mean(x_ref_original)

# Normalize
s = s / np.max(np.abs(s))
d = d / np.max(np.abs(d))
x_ref_original = x_ref_original / np.max(np.abs(x_ref_original))

n = np.arange(N)

print(f"Sampling frequency fs: {fs} Hz")
print(f"Nyquist frequency: {fs/2:.1f} Hz")

# --------------------------------------------------
# Create reference versions
# --------------------------------------------------

# Version 1: original reference
x_ref_unfiltered = x_ref_original.copy()

# Version 2: dampened reference
x_ref_dampened = damp_factor * x_ref_original.copy()

# Version 3: lowpass filtered + dampened reference
if cutoff >= fs / 2:
    raise ValueError("cutoff must be lower than Nyquist frequency fs/2")

b, a = butter(order, cutoff / (fs / 2), btype="low")
x_ref_lowpass = filtfilt(b, a, x_ref_original.copy())

# Normalize after filtering to avoid amplitude changes caused by filtering
x_ref_lowpass = x_ref_lowpass / np.max(np.abs(x_ref_lowpass))

# Dampen after filtering
x_ref_lowpass_dampened = damp_factor * x_ref_lowpass

# --------------------------------------------------
# NLMS function
# --------------------------------------------------

def nlms_filter(d, x_ref, L, beta, c):
    N = len(d)

    w = np.zeros(L)
    y = np.zeros(N)
    e = np.zeros(N)

    for i in range(L, N):

        x_vec = x_ref[i:i-L:-1]

        y[i] = np.dot(w, x_vec)

        e[i] = d[i] - y[i]

        norm_factor = c + np.dot(x_vec, x_vec)

        w = w + (beta / norm_factor) * x_vec * e[i]

    return y, e

# --------------------------------------------------
# SNR function
# --------------------------------------------------

def calculate_snr(s, d, e):
    noise_before = d - s
    noise_after = e - s

    snr_before = 10 * np.log10(
        np.sum(s**2) / np.sum(noise_before**2)
    )

    snr_after = 10 * np.log10(
        np.sum(s**2) / np.sum(noise_after**2)
    )

    snr_improvement = snr_after - snr_before

    return snr_before, snr_after, snr_improvement

# --------------------------------------------------
# Run tests
# --------------------------------------------------

tests = {
    "Unfiltered reference": x_ref_unfiltered,
    "Dampened reference": x_ref_dampened,
    "Lowpass + dampened reference": x_ref_lowpass_dampened
}

results = {}

for name, x_ref in tests.items():

    y, e = nlms_filter(d, x_ref, L, beta, c)

    snr_before, snr_after, snr_improvement = calculate_snr(s, d, e)

    actual_noise = d - s
    corr_noise_ref = np.corrcoef(actual_noise, x_ref)[0, 1]

    results[name] = {
        "x_ref": x_ref,
        "y": y,
        "e": e,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement": snr_improvement,
        "corr_noise_ref": corr_noise_ref
    }

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n--- Results ---")

for name, res in results.items():
    print(f"\n{name}")
    print(f"Correlation noise / x_ref: {res['corr_noise_ref']:.4f}")
    print(f"SNR before NLMS: {res['snr_before']:.2f} dB")
    print(f"SNR after NLMS: {res['snr_after']:.2f} dB")
    print(f"SNR improvement: {res['snr_improvement']:.2f} dB")
    print(f"Max abs x_ref: {np.max(np.abs(res['x_ref'])):.4f}")

# --------------------------------------------------
# Save best result
# --------------------------------------------------

os.makedirs("Audio files/Denoised", exist_ok=True)

best_name = max(results, key=lambda name: results[name]["snr_improvement"])
best_e = results[best_name]["e"]

sf.write("Audio files/Denoised/NLMS_best_test.wav", best_e, fs)

print(f"\nBest result: {best_name}")
print("File saved: Audio files/Denoised/NLMS_best_test.wav")

# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

fig, axs = plt.subplots(6, 1, figsize=(12, 14))

axs[0].plot(n, s)
axs[0].set_title("Clean Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(n, d)
axs[1].set_title("Desired Signal d(n): Noisy Speech")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(n, x_ref_unfiltered)
axs[2].set_title("Original Reference Signal x_ref(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(n, x_ref_dampened)
axs[3].set_title("Dampened Reference Signal 0.4 · x_ref(n)")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

axs[4].plot(n, x_ref_lowpass_dampened)
axs[4].set_title(f"Lowpass + Dampened Reference Signal, cutoff = {cutoff} Hz")
axs[4].set_ylabel("Amplitude")
axs[4].grid()

axs[5].plot(n, best_e)
axs[5].set_title(f"Best Cleaned Signal e(n): {best_name}")
axs[5].set_xlabel("Samples")
axs[5].set_ylabel("Amplitude")
axs[5].grid()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot cleaned signals from all tests
# --------------------------------------------------

plt.figure(figsize=(12, 6))

for name, res in results.items():
    plt.plot(n, res["e"], label=name, alpha=0.8)

plt.title("Comparison of Cleaned Signals e(n)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()