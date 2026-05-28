import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 8
beta = 0.005
c = 1e-6
max_duration = 10

# --------------------------------------------------
# Load desired signal d(n)
# --------------------------------------------------

d_file = "Audio files/With noise/Adaptiv/Mest tale.wav"
d, fs = librosa.load(d_file, sr=None, mono=True)

d = d[:int(max_duration * fs)]
d = d - np.mean(d)
d = d / np.max(np.abs(d))

# --------------------------------------------------
# Load reference signal x_ref(n)
# --------------------------------------------------

x_ref_file = "Audio files/With noise/Adaptiv/Mest støj.wav"
x_ref, fs_ref = librosa.load(x_ref_file, sr=fs, mono=True)

x_ref = x_ref[:len(d)]
x_ref = x_ref - np.mean(x_ref)
x_ref = x_ref / np.max(np.abs(x_ref))

# --------------------------------------------------
# Make sure signals have same length
# --------------------------------------------------

N = min(len(d), len(x_ref))

d = d[:N]
x_ref = x_ref[:N]

n = np.arange(N)

# --------------------------------------------------
# NLMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Error signal / cleaned speech

# --------------------------------------------------
# NLMS algorithm
# --------------------------------------------------

for i in range(L, N):

    x_vec = x_ref[i:i-L:-1]

    y[i] = np.dot(w, x_vec)

    e[i] = d[i] - y[i]

    norm_factor = c + np.dot(x_vec, x_vec)

    w = w + (beta / norm_factor) * x_vec * e[i]

# --------------------------------------------------
# Save as new file
# --------------------------------------------------

sf.write("Audio files/Denoised/NLMS_statinoary_finale.wav", e, fs) 
print("Filer gemt: NLMS_statinoary_finale.wav")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

axs[0].plot(n, d)
axs[0].set_title("Desired Signal d(n): Speech-Dominant Recording")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(n, x_ref)
axs[1].set_title("Reference Signal x_ref(n): Noise-Dominant Recording")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(n, y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(n, e)
axs[3].set_title("Cleaned Speech / Error Signal e(n)")
axs[3].set_xlabel("Samples")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()