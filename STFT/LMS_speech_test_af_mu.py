import numpy as np
import matplotlib.pyplot as plt
import librosa

# --------------------------------------------------
# Parameters
# --------------------------------------------------

L = 8
max_duration = 10

# Test different mu values
mu_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]

# --------------------------------------------------
# Load clean audio signal
# --------------------------------------------------

file_path = "Audio files/No noise/Mikkel_24gain.wav"
s, fs = librosa.load(file_path, sr=None, mono=True)

s = s[:int(max_duration * fs)]
s = s - np.mean(s)

N = len(s)
t = np.arange(N) / fs

# --------------------------------------------------
# Create non-stationary noise
# --------------------------------------------------

noise = (1 + 0.5*np.sin(2*np.pi*1*t)) * np.random.randn(N)

# Scale noise down a bit
noise = 0.05 * noise

# Noisy signal
d = s + noise

# Reference noise
x_ref = noise + 0.01*np.random.randn(N)

# --------------------------------------------------
# Upper bound for mu
# --------------------------------------------------

Px = np.mean(x_ref**2)

mu_max = 2 / (L * Px)
mu_conservative = 1 / (L * Px)

print("--------------------------------------------------")
print("Step-size stability estimate")
print("--------------------------------------------------")
print(f"Reference signal power Px = {Px:.6f}")
print(f"Theoretical upper bound mu_max = {mu_max:.4f}")
print(f"Conservative upper bound = {mu_conservative:.4f}")
print("--------------------------------------------------\n")

# --------------------------------------------------
# SNR before LMS
# --------------------------------------------------

noise_before = d - s

SNR_before = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_before**2)
)

# --------------------------------------------------
# LMS function
# --------------------------------------------------

def run_lms(mu):
    w = np.zeros(L)

    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(L, N):
        x_vec = x_ref[n:n-L:-1]

        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]

        w = w + mu * x_vec * e[n]

    return y, e, w

# --------------------------------------------------
# Run LMS for each mu value
# --------------------------------------------------

results = []

plt.figure(figsize=(12, 6))

window = int(0.05 * fs)   # 50 ms smoothing window

for mu in mu_values:
    y, e, w = run_lms(mu)

    noise_after = e - s

    SNR_after = 10 * np.log10(
        np.mean(s**2) / np.mean(noise_after**2)
    )

    SNR_improvement = SNR_after - SNR_before

    mse_smooth = np.convolve(
        e**2,
        np.ones(window) / window,
        mode='same'
    )

    results.append([mu, SNR_before, SNR_after, SNR_improvement])

    plt.plot(t, mse_smooth, label=f"mu = {mu}")

# --------------------------------------------------
# Plot MSE curves
# --------------------------------------------------

plt.title("Smoothed MSE Curves for Different Step Sizes")
plt.xlabel("Time [s]")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Print SNR table
# --------------------------------------------------

print("SNR comparison")
print("--------------------------------------------------")
print(f"{'mu':>10} {'SNR before':>15} {'SNR after':>15} {'Improvement':>15}")
print("--------------------------------------------------")

for row in results:
    print(f"{row[0]:>10.4f} {row[1]:>15.2f} {row[2]:>15.2f} {row[3]:>15.2f}")

print("--------------------------------------------------")

# --------------------------------------------------
# Plot SNR after LMS for each mu
# --------------------------------------------------

mu_plot = [row[0] for row in results]
snr_after_plot = [row[2] for row in results]

plt.figure(figsize=(10, 4))
plt.plot(mu_plot, snr_after_plot, marker='o')
plt.axvline(mu_max, linestyle='--', label=f"mu max = {mu_max:.3f}")

plt.title("SNR After LMS for Different Step Sizes")
plt.xlabel("Step size mu")
plt.ylabel("SNR after LMS [dB]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Run LMS with selected mu for signal plots
# --------------------------------------------------

mu_selected = 0.01

y, e, w = run_lms(mu_selected)

# --------------------------------------------------
# SNR for selected mu
# --------------------------------------------------

noise_after = e - s

SNR_after = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_after**2)
)

print(f"\nSelected mu = {mu_selected}")
print(f"SNR before LMS: {SNR_before:.2f} dB")
print(f"SNR after LMS: {SNR_after:.2f} dB")
print(f"SNR improvement: {SNR_after - SNR_before:.2f} dB")

# --------------------------------------------------
# Plot selected signal result
# --------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

axs[0].plot(t, s)
axs[0].set_title("Clean Audio Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

axs[1].plot(t, d)
axs[1].set_title(f"Noisy Audio Signal d(n) | SNR = {SNR_before:.2f} dB")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

axs[2].plot(t, y)
axs[2].set_title(f"Estimated Noise y(n), mu = {mu_selected}")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

axs[3].plot(t, e)
axs[3].set_title(f"Cleaned Signal e(n), mu = {mu_selected} | SNR = {SNR_after:.2f} dB")
axs[3].set_xlabel("Time [s]")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()