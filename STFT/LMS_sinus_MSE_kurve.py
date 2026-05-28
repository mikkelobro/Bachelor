import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Parameters
# --------------------------------------------------

fs = 1000
T = 2

N = int(fs * T)

L = 10
mu = 0.01

# --------------------------------------------------
# Sample and time axis
# --------------------------------------------------

n = np.arange(N)
t = n / fs

# --------------------------------------------------
# Clean sinus signal
# --------------------------------------------------

f0 = 10
s = np.sin(2*np.pi*f0*t)

# --------------------------------------------------
# Create non-stationary noise
# --------------------------------------------------

noise = 0.05 * np.random.randn(N)

# Noisy signal
d = s + noise

# Reference noise
x_ref = noise + 0.01*np.random.randn(N)


# --------------------------------------------------
# Stability limit for LMS step size
# --------------------------------------------------
p = L - 1 

Ex2 = np.mean(x_ref**2)

mu_max = 2 / ((p + 1) * Ex2)

print(f"p = {p}")
print(f"E[x^2(n)] = {Ex2:.6f}")
print(f"mu must satisfy: 0 < mu < {mu_max:.6f}")

# --------------------------------------------------
# Compare different mu values
# --------------------------------------------------

mu_values = [0.001, 0.01, 0.05, 0.1]

plt.figure(figsize=(10,6))

for mu_test in mu_values:

    # Initialize filter
    w_test = np.zeros(L)

    y_test = np.zeros(N)
    e_test = np.zeros(N)

    # LMS algorithm
    for i in range(L, N):

        x_vec = x_ref[i:i-L:-1]

        y_test[i] = np.dot(w_test, x_vec)

        e_test[i] = d[i] - y_test[i]

        w_test = w_test + mu_test * x_vec * e_test[i]

    # --------------------------------------------------
    # Instantaneous squared error
    # --------------------------------------------------

    error_power = (s - e_test)**2

    # Avoid log(0)
    error_power = error_power + 1e-12

    # Convert to dB
    mse_db = 10 * np.log10(error_power)

    # Smooth curve
    window = 10

    mse_db_smooth = np.convolve(
        mse_db,
        np.ones(window)/window,
        mode='same'
    )

    # Plot
    plt.plot(n, mse_db_smooth, label=f"$\mu$ = {mu_test}")

# --------------------------------------------------
# Plot settings
# --------------------------------------------------

plt.xlabel("Number of iterations, k")
plt.ylabel("MSE [dB]")

plt.title("Learning Curve for LMS")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot settings
# --------------------------------------------------

plt.xlabel("Samples")
plt.ylabel("MSE")
plt.title("MSE Convergence for Different Step Sizes")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# --------------------------------------------------
# Run LMS with selected mu
# --------------------------------------------------

w = np.zeros(L)

y = np.zeros(N)
e = np.zeros(N)

for i in range(L, N):

    x_vec = x_ref[i:i-L:-1]

    y[i] = np.dot(w, x_vec)

    e[i] = d[i] - y[i]

    w = w + mu * x_vec * e[i]
