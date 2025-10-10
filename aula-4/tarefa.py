import numpy as np
import matplotlib.pyplot as plt


def moving_average_kernel(k: int) -> np.ndarray:
    return np.ones(k, dtype=float) / float(k)


def stem(ax, n: np.ndarray, x: np.ndarray, title: str):
    ax.stem(n, x)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", linewidth=1, alpha=0.2)


def main():
    k = 8
    h = moving_average_kernel(k)

    # Inputs
    impulse_len = 16
    step_len = 32
    x_vec = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)

    impulse = np.zeros(impulse_len, dtype=float)
    impulse[0] = 1.0
    step = np.ones(step_len, dtype=float)

    # Convolutions (full)
    y_impulse = np.convolve(impulse, h, mode="full")
    y_step = np.convolve(step, h, mode="full")
    y_x = np.convolve(x_vec, h, mode="full")

    # Print results
    np.set_printoptions(precision=4, suppress=True)
    print("k =", k)
    print("h[n] (impulse response):", h)
    print(
        "\nInput: unit impulse (length =",
        impulse_len,
        ")\nOutput y_impulse[n]:\n",
        y_impulse,
    )
    print("\nInput: unit step (length =", step_len, ")\nOutput y_step[n]:\n", y_step)
    print("\nInput: x[n] =", x_vec, "\nOutput y_x[n]:\n", y_x)

    # Plots
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False)

    # Impulse and its output
    n_impulse = np.arange(len(impulse))
    n_y_impulse = np.arange(len(y_impulse))
    stem(axes[0, 0], n_impulse, impulse, "Impulso unitário x[n]")
    stem(axes[0, 1], n_y_impulse, y_impulse, "Saída y[n] para impulso")

    # Step and its output
    n_step = np.arange(len(step))
    n_y_step = np.arange(len(y_step))
    stem(axes[1, 0], n_step, step, "Degrau unitário x[n]")
    stem(axes[1, 1], n_y_step, y_step, "Saída y[n] para degrau")

    # x_vec and its output
    n_x = np.arange(len(x_vec))
    n_y_x = np.arange(len(y_x))
    stem(axes[2, 0], n_x, x_vec, "x[n] = [1, 0.5, 0.25, 0.125]")
    stem(axes[2, 1], n_y_x, y_x, "Saída y[n] para x[n]")

    fig.tight_layout()
    plt.savefig(
        "/Users/alexandre/www/univali/processamento-digital-de-sinais/aula-4/tarefa_outputs.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
