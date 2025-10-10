import numpy as np
import matplotlib.pyplot as plt


def plot_convolution():
    """Plot two discrete-time signals and their convolution using np.convolve."""
    # Define two simple discrete signals (finite-length sequences)
    # x[n]: a length-5 rectangular pulse
    input_signal = np.ones(5, dtype=float)

    # h[n]: a simple 3-point moving-average kernel
    impulse_response = np.array([1.0, 1.0, 1.0]) / 3.0

    # Convolution (full)
    output_signal = np.convolve(input_signal, impulse_response, mode="full")

    # Sample indices for each sequence
    n_x = np.arange(len(input_signal))
    n_h = np.arange(len(impulse_response))
    n_y = np.arange(len(output_signal))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

    axes[0].stem(n_x, input_signal)
    axes[0].set_title("Input x[n]")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(n_h, impulse_response)
    axes[1].set_title("Impulse response h[n]")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(n_y, output_signal)
    axes[2].set_title("Convolution y[n] = x[n] * h[n]")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    # Optionally save the figure next to the script
    plt.savefig("convolution.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_convolution()
