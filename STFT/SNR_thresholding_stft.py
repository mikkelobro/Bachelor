import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

# --- Load noisy audio file ---
file_path = "Audio files/With noise/noisy_nonstationary.wav"
x, fs = librosa.load(file_path, sr=None, mono=True)

# --- Load clean reference audio file ---
clean_path = "Audio files/No noise/Mikkel_24år.wav"
s, fs_clean = librosa.load(clean_path, sr=None, mono=True)

# --- Check sampling frequencies ---
if fs != fs_clean:
    raise ValueError("Clean and noisy signals must have the same sampling frequency")

# --- Use only the first 10 seconds ---
max_duration = 10
max_samples = int(max_duration * fs)

x = x[:max_samples]
s = s[:max_samples]

# --- Make signals same length ---
min_len = min(len(x), len(s))
x = x[:min_len]
s = s[:min_len]

# --- Remove DC-offset ---
x = x - np.mean(x)
s = s - np.mean(s)

# --- Time axis ---
t = np.arange(len(x)) / fs

# --- STFT parameters ---
win_size = 1024
hop = 256
window = np.hanning(win_size)

# --- STFT of noisy signal ---
stft_frames = []

for i in range(0, len(x) - win_size, hop):
    frame = x[i:i + win_size]
    frame = frame * window
    X = np.fft.rfft(frame)
    stft_frames.append(X)

stft_frames = np.array(stft_frames).T

# --- Magnitude and phase ---
mag = np.abs(stft_frames)
phase = np.angle(stft_frames)

# --- Global thresholding ---
threshold_factor = 0.01
threshold = threshold_factor * np.max(mag)

mask = mag >= threshold
mag_clean = mag * mask

# --- Reconstruct complex STFT ---
stft_clean = mag_clean * np.exp(1j * phase)

# --- Inverse STFT using overlap-add ---
x_clean = np.zeros(len(x))
window_sum = np.zeros(len(x))

frame_idx = 0

for i in range(0, len(x) - win_size, hop):
    frame_time = np.fft.irfft(stft_clean[:, frame_idx], n=win_size)
    x_clean[i:i + win_size] += frame_time * window
    window_sum[i:i + win_size] += window**2
    frame_idx += 1

valid = window_sum > 1e-8
x_clean[valid] /= window_sum[valid]

# --- Calculate SNR before and after denoising ---
def calculate_snr(clean_signal, test_signal):
    min_len = min(len(clean_signal), len(test_signal))
    clean_signal = clean_signal[:min_len]
    test_signal = test_signal[:min_len]

    noise = test_signal - clean_signal

    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)

    return 10 * np.log10(signal_power / noise_power)

snr_before = calculate_snr(s, x)
snr_after = calculate_snr(s, x_clean)

print(f"SNR before denoising: {snr_before:.2f} dB")
print(f"SNR after denoising: {snr_after:.2f} dB")
print(f"SNR improvement: {snr_after - snr_before:.2f} dB")

# --- Normalise only for saving/listening ---
x_clean_save = x_clean / np.max(np.abs(x_clean))

# --- Save denoised audio ---
sf.write("Audio files/Denoised/threshold_cleaned_output.wav", x_clean_save, fs)

print("File saved: threshold_cleaned_output.wav")

# --- Axes for spectrogram ---
freqs = np.fft.rfftfreq(win_size, d=1/fs)
times = np.arange(mag.shape[1]) * hop / fs

# --- Plot 1: time signal ---
plt.figure()
plt.plot(t, x, alpha=0.6, label="Noisy audio")
plt.plot(t, x_clean, alpha=0.7, label="Thresholded audio")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Noisy vs thresholded audio")
plt.legend()
plt.grid()
plt.show()

# --- Plot 2: spectrogram before thresholding ---
plt.figure()
plt.pcolormesh(times, freqs, 20*np.log10(mag + 1e-8), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram before thresholding")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()

# --- Plot 3: spectrogram after thresholding ---
plt.figure()
plt.pcolormesh(times, freqs, 20*np.log10(mag_clean + 1e-8), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram after thresholding")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()