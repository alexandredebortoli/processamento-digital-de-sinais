import numpy as np
import matplotlib.pyplot as plt


def impulse_signal(N):
    """Impulso unitário"""
    x = np.zeros(N)
    x[0] = 1
    return x


def step_signal(N):
    """Degrau unitário"""
    return np.ones(N)


def compute_output(R, x, N):
    """
    Calcula y[n]:
    y[n] = x[n] - x[n-1] + R*y[n-1]
    """
    y = np.zeros(N)
    for n in range(N):
        xn = x[n] if n < len(x) else 0
        xn1 = x[n - 1] if n - 1 >= 0 else 0
        yn1 = y[n - 1] if n - 1 >= 0 else 0
        y[n] = xn - xn1 + R * yn1
    return y


def plot_signal(n, y, title):
    """Plota o sinal y[n]"""
    plt.stem(n, y, basefmt="k-")
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    R = 0.95
    N = 30
    n = np.arange(N)

    # a. Saída para entrada impulso
    x_imp = impulse_signal(N)
    y_imp = compute_output(R, x_imp, N)
    print("Saída para impulso:", y_imp)
    plot_signal(n, y_imp, "Resposta ao Impulso")

    # b. Saída para entrada degrau
    x_step = step_signal(N)
    y_step = compute_output(R, x_step, N)
    print("Saída para degrau:", y_step)
    plot_signal(n, y_step, "Resposta ao Degrau")
