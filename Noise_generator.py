import numpy as np
from scipy.io import wavfile

# Upload file
fs, x = wavfile.read("Audio files/No noise/Mikkel_24år.wav")

# Convert to mono
if x.ndim > 1:
    x = np.mean(x, axis=1)

# Normalize
x = x / np.max(np.abs(x))

# Apply noise
t = np.linspace(0, 1, len(x)) 
f = 5
A = 0.5
noise_level = 0.05
noise_stat = noise_level * np.random.randn(len(x))
x_noisy_stat = x + noise_stat

noise_nonstat = noise_level * np.random.randn(len(x))  * (A + 1 + np.sin(f * np.pi * t))
x_noisy_nonstat = x + noise_nonstat

# Length of noise-only files in seconds
duration = 20

# Number of samples
N = int(fs * duration)

# Repeat noise if necessary
noise_stat_long = np.tile(noise_stat, int(np.ceil(N / len(noise_stat))))[:N]
noise_nonstat_long = np.tile(noise_nonstat, int(np.ceil(N / len(noise_nonstat))))[:N]

# Save noise-only signals
noise_out_stat = np.int16(noise_stat_long / np.max(np.abs(noise_stat_long)) * 32767)
wavfile.write("Audio files/With noise/noise_stationary_only.wav", fs, noise_out_stat)

noise_out_nonstat = np.int16(noise_nonstat_long / np.max(np.abs(noise_nonstat_long)) * 32767)
wavfile.write("Audio files/With noise/noise_nonstationary_only.wav", fs, noise_out_nonstat)


x_out_stat = np.int16(x_noisy_stat / np.max(np.abs(x_noisy_stat)) * 32767)
wavfile.write("Audio files/With noise/noisy_stationary.wav", fs, x_out_stat)

x_out_nonstat = np.int16(x_noisy_nonstat / np.max(np.abs(x_noisy_nonstat)) * 32767)
wavfile.write("Audio files/With noise/noisy_nonstationary.wav", fs, x_out_nonstat)