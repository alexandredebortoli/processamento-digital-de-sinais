import numpy as np
import matplotlib.pyplot as plt


def unit_step(n):
    """Degrau unitário"""
    return np.where(n >= 0, 1, 0)


def impulse(n):
    """Impulso unitário"""
    return np.where(n == 0, 1, 0)


def h_sequence(n):
    """Resposta ao impulso h[n] = (0.5)^n * u[n]"""
    return (0.5) ** n * unit_step(n)


def x_sequence(n):
    """x[n] = u[n] - u[n-2]"""
    return unit_step(n) - unit_step(n - 2)


def convolution(x, h):
    """Convolução discreta entre x e h"""
    return np.convolve(x, h)


def plot_signal(n, y, title):
    plt.stem(n, y, basefmt="k-")
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    N = 9  # 0<= n <= 8
    n = np.arange(0, N)

    # a. Saída para x[n] = u[n]-u[n-2]
    x = x_sequence(n)
    h = h_sequence(n)
    y = convolution(x, h)[:N]
    print("Valores de y[n] para 0 ≤ n ≤ 8:", y)

    plot_signal(np.arange(len(y)), y, "Saída y[n] para x[n] = u[n]-u[n-2]")

    # b. Saída para impulso unitário
    x_imp = impulse(n)
    y_imp = convolution(x_imp, h)[:N]
    print("Resposta ao impulso (h[n]) para 0 ≤ n ≤ 8:", y_imp)

    plot_signal(np.arange(len(y_imp)), y_imp, "Resposta ao Impulso h[n]")
