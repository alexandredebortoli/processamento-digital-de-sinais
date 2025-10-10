import numpy as np
import matplotlib.pyplot as plt


def plot_convolution():
    """Plot two discrete-time signals and their convolution using np.convolve."""
    # Define two simple discrete signals (finite-length sequences)
    # x[n]: a length-5 rectangular pulse
    x = np.ones(6, dtype=float)

    # h[n]
    h = np.array([1.0, 0.5, 0.25, 0.125])

    # Convolution (full)
    y = np.convolve(x, h, mode="full")

    # Sample indices for each sequence
    n_x = np.arange(len(x))
    n_h = np.arange(len(h))
    n_y = np.arange(len(y))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

    axes[0].stem(n_x, x)
    axes[0].set_title("Input x[n]")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(n_h, h)
    axes[1].set_title("Impulse response h[n]")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(n_y, y)
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
