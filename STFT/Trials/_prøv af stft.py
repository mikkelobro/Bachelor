import numpy as np
import matplotlib.pyplot as plt

# --- Signal ---
fs = 1000
T = 10
t = np.arange(0, T, 1/fs)

f0, f1 = 1, 500
chirp_rate = (f1 - f0)/T
signal = np.sin(2*np.pi*(f0*t + 0.5*chirp_rate*t**2))

noise = 0.2*np.random.randn(len(t))
x = signal + noise

# --- Vælg time-segment ---
N = 32
win_len = len(x)//N

segment_id = 10  # skift denne: 0,1,2,3,4
seg_start = segment_id * win_len
seg_end = seg_start + win_len

x_win = x[seg_start:seg_end]
t_win = t[seg_start:seg_end]

# --- STFT parametre ---
win_size = 256
hop = 128
window = np.hanning(win_size)

# --- 1) Lav STFT kun på segmentet ---
stft_seg = []
for i in range(0, len(x_win) - win_size, hop):
    frame = x_win[i:i+win_size]
    frame = frame - np.mean(frame)
    frame = frame * window
    X = np.fft.rfft(frame)
    stft_seg.append(np.abs(X))

stft_seg = np.array(stft_seg).T  # shape: [freq_bins, time_frames_seg]

# --- 2) Lav en "tom" STFT for hele signalet (samme tids-grid som en fuld STFT ville have) ---
num_frames_full = 1 + (len(x) - win_size) // hop
freq_bins = stft_seg.shape[0]

stft_full = np.full((freq_bins, num_frames_full), np.nan)  # NaN = tomt/ingen data

# Find hvor segmentets frames starter i fuld STFT-tidsindeks
start_frame_full = seg_start // hop
num_frames_seg = stft_seg.shape[1]

# Indsæt segment-data i den store matrix
end_frame_full = min(start_frame_full + num_frames_seg, num_frames_full)
stft_full[:, start_frame_full:end_frame_full] = stft_seg[:, :end_frame_full-start_frame_full]

# --- Akser til plot ---
freqs = np.fft.rfftfreq(win_size, d=1/fs)
times_full = np.arange(num_frames_full) * hop / fs

# --- Plot: signal + valgt segment ---
plt.figure()
plt.plot(t, x, alpha=0.3, label="Whole signal")
plt.plot(t_win, x_win, label="Chosen time segment")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Chirp signal with gaussian noise")
plt.show()

plt.figure()

freqs = np.fft.rfftfreq(win_size, d=1/fs)

for i in range(0, len(x_win) - win_size, hop):

    frame = x_win[i:i+win_size]
    frame = frame - np.mean(frame)
    frame = frame * window

    X = np.fft.rfft(frame)
    mag = np.abs(X)

    # tidspunktet for vinduet
    frame_time = t_win[0] + (i + win_size/2)/fs

    plt.plot(freqs, mag, alpha=0.3)

plt.xlim(0,500)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT magnitude for all windows in the chosen segment")
plt.grid()

plt.show()

# --- Plot: spectrogram med kun segment-data, men x-akse 0..10 ---
plt.figure()
plt.pcolormesh(times_full, freqs, stft_full, shading="auto")
plt.ylim(0, 500)
plt.xlim(0, T)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude")
plt.title("Spectrogram (only chosen segment shown)")
plt.show()