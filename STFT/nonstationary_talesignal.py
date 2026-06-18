import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

# --- Load audio file ---
file_path = "Audio files/With noise/noisy_nonstationary.wav"
#file_path = "Audio files/With noise/Bil.wav"
x, fs = librosa.load(file_path, sr=None, mono=True)

# --- Load clean reference signal ---
clean_path = "Audio files/No noise/Mikkel_24år.wav"
s, fs_clean = librosa.load(clean_path, sr=None, mono=True)

# Sørg for samme længde
s = s[:len(x)]
x = x[:len(s)]

# Fjern DC-offset fra clean signal
s = s - np.mean(s)

# --- Brug kun de første 10 sekunder hvis ønsket ---
max_duration = 10
x = x[:int(max_duration * fs)]

# --- Fjern DC-offset ---
x = x - np.mean(x)

# --- Tidsakse ---
t = np.arange(len(x)) / fs
T = len(x) / fs

# --- STFT parametre ---
win_size = 1024
hop = 256
window = np.hanning(win_size)

# --- Lav STFT af hele signalet ---
stft_frames = []
for i in range(0, len(x) - win_size, hop):
    frame = x[i:i + win_size]
    frame = frame * window
    X = np.fft.rfft(frame)
    stft_frames.append(X)

stft_frames = np.array(stft_frames).T   # [freq_bins, time_frames]

# --- Magnitude og fase ---
mag = np.abs(stft_frames)
phase = np.angle(stft_frames)

# --- Thresholding ---
threshold_factor = 0.1  # prøv fx 0.05, 0.1, 0.2
threshold = threshold_factor * np.max(mag)

mask = mag >= threshold
mag_clean = mag * mask

# --- Rekonstruér kompleks STFT ---
stft_clean = mag_clean * np.exp(1j * phase)

# --- Inverse STFT med overlap-add ---
x_clean = np.zeros(len(x))
window_sum = np.zeros(len(x))

frame_idx = 0
for i in range(0, len(x) - win_size + 1, hop):
    frame_time = np.fft.irfft(stft_clean[:, frame_idx], n=win_size)
    x_clean[i:i + win_size] += frame_time * window
    window_sum[i:i + win_size] += window**2
    frame_idx += 1

valid = window_sum > 1e-8
x_clean[valid] /= window_sum[valid]

# --- Normaliser signalet ---
x_clean = x_clean / np.max(np.abs(x_clean))


# --- Normaliser alle signaler på samme måde før SNR ---
def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal

#Sørg for samme længde
min_len = min(len(s), len(x), len(x_clean))

s_snr = s[:min_len]
x_snr = x[:min_len]
x_clean_snr = x_clean[:min_len]

## Fjern DC-offset fra alle signaler
s_snr = s_snr - np.mean(s_snr)
x_snr = x_snr - np.mean(x_snr)
x_clean_snr = x_clean_snr - np.mean(x_clean_snr)

# Normaliser alle signaler ens
s_snr = normalize_signal(s_snr)
x_snr = normalize_signal(x_snr)
x_clean_snr = normalize_signal(x_clean_snr)

# --- Beregn SNR før og efter denoising ---
def calculate_snr(clean_signal, test_signal):
    noise = test_signal - clean_signal
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
       return np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_before = calculate_snr(s_snr, x_snr)
snr_after = calculate_snr(s_snr, x_clean_snr)

print(f"SNR before denoising: {snr_before:.2f} dB")
print(f"SNR after denoising: {snr_after:.2f} dB")
print(f"SNR improvement: {snr_after - snr_before:.2f} dB")

# Ekstra kontrol
noise_before = x_snr - s_snr
noise_after = x_clean_snr - s_snr

print(f"Clean signal power: {np.mean(s_snr**2):.6f}")
print(f"Noise power before: {np.mean(noise_before**2):.6f}")
print(f"Noise power after: {np.mean(noise_after**2):.6f}")

# --- Gem lydfiler ---
sf.write("Audio files/Denoised/threshold_cleaned_output.wav", x_clean, fs)

print("Filer gemt: threshold_cleaned_output.wav")

# --- Akser til spectrogram ---
freqs = np.fft.rfftfreq(win_size, d=1/fs)
times = np.arange(mag.shape[1]) * hop / fs

# --- Plot 1: tidssignal ---
plt.figure()
plt.plot(t, x, alpha=0.6, label="Noisy audio")
plt.plot(t, x_clean, alpha=0.7, label="Thresholded audio")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Noisy vs thresholded audio")
plt.legend()
plt.grid()
plt.show()

# --- Plot 2: spectrogram før ---
plt.figure()
plt.pcolormesh(times, freqs, 20*np.log10(mag + 1e-8), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram before thresholding")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()

# --- Plot 3: spectrogram efter ---
plt.figure()
plt.pcolormesh(times, freqs, 20*np.log10(mag_clean + 1e-8), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram after thresholding")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()