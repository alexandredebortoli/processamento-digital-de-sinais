import numpy as np
import matplotlib.pyplot as plt


def compute_output(a0, a1, x, N):
    """
    Calcula a saída y[n]:
    y[n] = a0*x[n] + a1*y[n-1]
    """
    y = np.zeros(N)
    for n in range(N):
        xn = x[n] if n < len(x) else 0
        yn1 = y[n - 1] if n > 0 else 0
        y[n] = a0 * xn + a1 * yn1
    return y


def impulse_signal(N):
    """Gera um impulso unitário"""
    x = np.zeros(N)
    x[0] = 1
    return x


def step_signal(N):
    """Gera um degrau unitário"""
    return np.ones(N)


def plot_signal(n, y, title):
    """Plota o sinal y[n]"""
    plt.stem(n, y, basefmt="k-")
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    a0 = 1.0
    a1 = 0.8
    N = 20

    n = np.arange(N)

    # Entrada impulso
    x_imp = impulse_signal(N)
    y_imp = compute_output(a0, a1, x_imp, N)
    plot_signal(n, y_imp, "Resposta ao Impulso")

    # Entrada degrau
    x_step = step_signal(N)
    y_step = compute_output(a0, a1, x_step, N)
    plot_signal(n, y_step, "Resposta ao Degrau")
