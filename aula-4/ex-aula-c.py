import numpy as np
import matplotlib.pyplot as plt


def plot_convolution():
    """Plot two discrete-time signals and their convolution using np.convolve."""
    # Define two simple discrete signals (finite-length sequences)
    # x[n]: a length-5 rectangular pulse

    X_ZERO_INDEX = 2
    H_ZERO_INDEX = 0

    # C)
    x = np.array([-1, -1, 0, 1, 1])

    # h[n]
    h = np.array([1, 0.5, 0.25, 0.125, 0, 0])

    # Convolution (full)
    y = np.convolve(x, h, mode="full")

    # Sample indices for each sequence adjusted so that index 0 is at the
    # specified zero index positions
    n_x = np.arange(len(x)) - X_ZERO_INDEX
    n_h = np.arange(len(h)) - H_ZERO_INDEX
    # For convolution, the zero index is the sum of the individual zero indices
    n_y = np.arange(len(y)) - (X_ZERO_INDEX + H_ZERO_INDEX)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

    axes[0].stem(n_x, x)
    axes[0].set_title("Input x[n]")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color="k", linewidth=1, alpha=0.3)
    axes[0].set_xlim(n_x.min() - 1, n_x.max() + 1)

    axes[1].stem(n_h, h)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color="k", linewidth=1, alpha=0.3)
    axes[1].set_xlim(n_h.min() - 1, n_h.max() + 1)

    axes[2].stem(n_y, y)
    axes[2].set_title("Convolution y[n] = x[n] * h[n]")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(0, color="k", linewidth=1, alpha=0.3)
    axes[2].set_xlim(n_y.min() - 1, n_y.max() + 1)

    fig.tight_layout()
    # Optionally save the figure next to the script
    plt.savefig("convolution.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_convolution()
