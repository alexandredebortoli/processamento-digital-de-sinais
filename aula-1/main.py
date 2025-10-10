import numpy as np
import matplotlib.pyplot as plt


def main():
    plt.close("all")

    # Cria o sinal contínuo
    Dt = 5e-5
    t = np.arange(-5e-3, 5e-3 + Dt, Dt)
    f = 1000
    xa = np.cos(2 * np.pi * f * t)

    # Mostra o sinal contínuo
    plt.subplot(2, 1, 1)
    plt.plot(t * 1000, xa)
    plt.xlabel("Tempo [ms]")
    plt.ylabel("xa(t)")
    plt.title(f"Sinal de Freq = {f}Hz")
    plt.grid(True)

    # Amostra o sinal
    Fs = 8000
    Ts = 1 / Fs
    n = np.arange(-20, 21)
    xd = np.cos(2 * np.pi * n * f / Fs)

    # Marca os pontos de amostragem
    plt.plot(n * Ts * 1000, xd, "ro", markersize=4)

    # Mostra só as amostras
    plt.subplot(2, 1, 2)
    plt.stem(n, xd)
    plt.title("Sinal amostrado")
    plt.xlabel("Amostras")
    plt.ylabel("x[n]")
    plt.grid(True)

    plt.tight_layout()

    # Salva o gráfico
    plt.savefig("amostragem.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo!")

    plt.show()


if __name__ == "__main__":
    main()
