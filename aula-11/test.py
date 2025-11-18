import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def moving_average_processing(sample_vector, coefficient_vector):
    """
    Simple moving average system: y[n] = a0* x[n] + a1* x[n-1] + a2* x[n-2] + a3* x[n-3]
    """
    output = 0
    for n in range(0, len(sample_vector)):
        output += coefficient_vector[n] * sample_vector[n]
    return output


def update_w(w, e, sample_vector, mu):
    for i in range(len(w)):
        w[i] = w[i] + mu * e * sample_vector[i]
    return w


def process_lms(x, K):
    MU = 0.001
    coefficient_vector = np.array([1 / K] * K)
    sample_vector = np.zeros(K)
    w = np.zeros(K)
    e = np.zeros(len(x))

    for n in range(len(x)):
        # Update sample vector (shift and add new sample)
        sample_vector[1:] = sample_vector[:-1]
        sample_vector[0] = x[n]

        # Calculate desired output (moving average filter output)
        d = moving_average_processing(sample_vector, coefficient_vector)
        print(f"d[{n}] = {d}")

        # Calculate adaptive filter output (dot product)
        y = convolve(sample_vector, w)
        print(f"y[{n}] = {y}")

        # Calculate error
        e[n] = d - y
        print(f"e[{n}] = {e[n]}")

        # Update filter coefficients using LMS algorithm
        w = update_w(w, e[n], sample_vector, MU)

    return e


def main():
    try:
        x = np.fromfile("white-noise.pcm", dtype=np.int16)
        print(f"📁 Loaded PCM file: {len(x)} samples")
    except FileNotFoundError:
        print("❌ Error: white-noise.pcm not found!")
        return

    NUM_COEFFICIENTS = 8
    K_VALUE = 8

    e = process_lms(x, K_VALUE)

    # Calculate statistics
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
