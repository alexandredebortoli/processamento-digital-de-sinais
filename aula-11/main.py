# Filtro Adaptativo - LMS
# X(n) = [1, 2, 3, 4, 5]
# d = a0 * x(n) + a1 * x(n-1) + a2 * x(n-2) ... + a7 * x(n-7)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


# SISTEMA "DESCONHECIDO"
def system_processing(x: np.array, K: int = 8):
    def moving_average_processing(sample_vector, coefficient_vector):
        """
        Simple moving average system: y[n] = a0* x[n] + a1* x[n-1] + a2* x[n-2] + a3* x[n-3]
        """
        output = 0
        for n in range(0, len(sample_vector)):
            output += coefficient_vector[n] * sample_vector[n]
        return output

    # x = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # K = 10
    coefficient_vector = np.array([1 / K] * K)
    sample_vector = np.zeros(K)
    y = np.zeros(len(x))
    for n in range(len(x)):
        sample_vector[1:] = sample_vector[:-1]
        sample_vector[0] = x[n]

        y[n] = moving_average_processing(sample_vector, coefficient_vector)

    return y


def update_w(w, e, sample_vector, mu):
    for i in range(len(w)):
        w[i] = w[i] + mu * e[i] * sample_vector[i]
    return w


def main():
    NUM_COEFFICIENTS = 8
    MU = 0.001
    x = np.zeros(NUM_COEFFICIENTS)
    w = np.zeros(NUM_COEFFICIENTS)
    y = np.zeros(NUM_COEFFICIENTS)
    e = np.zeros(NUM_COEFFICIENTS)

    try:
        sample = np.fromfile("white-noise.pcm", dtype=np.int16)
        print(f"📁 Loaded PCM file: {len(x)} samples")
    except FileNotFoundError:
        print("❌ Error: white-noise.pcm not found!")
        return

    for n in range(len(sample)):
        x[1:] = x[:-1]
        x[0] = sample[n]
        d = system_processing(x)
        y = x * w
        e = d - y
        w = update_w(w, e, x, MU)

    print(f"📊 LMS Filter Statistics:")
    print(f"   Max Error: {np.max(np.abs(e)):.2f}")
    print(f"   Min Error: {np.min(np.abs(e)):.2f}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle(
        "LMS Adaptive Filter - Error Signal Analysis", fontsize=16, fontweight="bold"
    )

    # Plot 1: Full error signal
    axes.plot(e, "b-", linewidth=0.5, alpha=0.7)
    axes.set_title("Error Signal - Full View", fontweight="bold")
    axes.set_xlabel("Sample Index (n)")
    axes.set_ylabel("Error e[n]")
    axes.grid(True, alpha=0.3)
    axes.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
