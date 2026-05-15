import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Parameters
# --------------------------------------------------

N = 2000
L = 100
mu = 0.0005

# --------------------------------------------------
# Signals
# --------------------------------------------------

t = np.arange(N)

# Clean sinus signal
s = np.sin(2*np.pi*0.01*t)  #ren sinus

# Non-stationary noise
noise = (1 + 0.5*np.sin(2*np.pi*0.001*t)) * np.random.randn(N) #hvid gaussian støj på et signal med langsomt ændrende amplitude

# Desired signal (signal + noise)
d = s + noise

# Reference noise
x = noise + 0.1*np.random.randn(N) #støjen + ekstra hvid gaussisk støj med lavere amplitude

# --------------------------------------------------
# LMS initialization
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)   # Estimated noise
e = np.zeros(N)   # Cleaned signal

# --------------------------------------------------
# LMS algorithm
# --------------------------------------------------

for n in range(L, N):

    x_vec = x[n:n-L:-1]

    # Filter output
    y[n] = np.dot(w, x_vec)

    # Error signal
    e[n] = d[n] - y[n]

    # Update weights
    w = w + mu * x_vec * e[n]

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
# --------------------------------------------------
# SNR before LMS
# --------------------------------------------------

noise_before = d - s

SNR_before = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_before**2)
)

# --------------------------------------------------
# SNR after LMS
# --------------------------------------------------

noise_after = e - s

SNR_after = 10 * np.log10(
    np.mean(s**2) / np.mean(noise_after**2)
)

print(f"SNR before LMS: {SNR_before:.2f} dB")
print(f"SNR after LMS: {SNR_after:.2f} dB")


fig, axs = plt.subplots(4, 1, figsize=(12, 10))

# Plot 1
axs[0].plot(s)
axs[0].set_title("Original Clean Signal s(n)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Plot 2
axs[1].plot(d)
axs[1].set_title("Noisy Signal d(n)")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

# Plot 3
axs[2].plot(y)
axs[2].set_title("Estimated Noise y(n)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()

# Plot 4
axs[3].plot(e)
axs[3].set_title("Cleaned Signal e(n)")
axs[3].set_xlabel("Sample n")
axs[3].set_ylabel("Amplitude")
axs[3].grid()

plt.tight_layout()
plt.show()