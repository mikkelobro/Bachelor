import numpy as np
import matplotlib.pyplot as plt
import librosa

# --- Load audio file ---
file_path = "Audio files/With noise/noisy.wav"
x, fs = librosa.load(file_path, sr=None, mono=True)

# --- Create time axis ---
t = np.arange(len(x)) / fs

# --- Plot signal ---
plt.figure(figsize=(10, 4))
plt.plot(t, x, color='skyblue') 
plt.plot(t, x)
plt.title("Audio Signal in Time Domain with Noise")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()