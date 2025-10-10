import numpy as np
import matplotlib.pyplot as plt


def main():
    plt.close("all")

    # Parâmetros do eco
    Fs = 8000  # taxa de amostragem
    Ts = 1 / Fs  # período de amostragem (125us)
    D = 1e-3  # atraso de 1ms
    delay_samples = int(D * Fs)  # 8 amostras

    # Ganhos
    a0 = 1.0
    a1 = 0.5

    # Buffer de delay
    buffer_delay = np.zeros(delay_samples)

    # Entrada: impulso unitário
    entrada = np.zeros(50)  # vetor maior para ver o eco
    entrada[0] = 1  # impulso em n=0

    # Saída
    saida = np.zeros(len(entrada))

    # Aplica o eco
    for i in range(len(entrada)):
        # Entrada atual
        x = entrada[i]

        # Saída = sinal original + sinal atrasado
        y = a0 * x + a1 * buffer_delay[-1]

        # Atualiza o buffer de delay
        buffer_delay[1:] = buffer_delay[:-1]
        buffer_delay[0] = x

        saida[i] = y

    # Mostra os resultados
    plt.figure(figsize=(12, 8))

    # Impulso de entrada
    plt.subplot(3, 1, 1)
    plt.stem(entrada)
    plt.title("Entrada: Impulso unitário")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Resposta do eco
    plt.subplot(3, 1, 2)
    plt.stem(saida)
    plt.title("Saída: Resposta ao impulso")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Comparação
    plt.subplot(3, 1, 3)
    plt.stem(entrada, label="Entrada")
    plt.stem(saida, label="Saída")
    plt.title("Comparação")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Salva o gráfico
    plt.savefig("eco_impulso.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo!")

    plt.show()

    # Mostra informações
    print(f"Taxa de amostragem: {Fs} Hz")
    print(f"Período de amostragem: {Ts*1e6:.1f} μs")
    print(f"Atraso: {D*1000:.1f} ms")
    print(f"Atraso em amostras: {delay_samples}")
    print(f"Ganho original: {a0}")
    print(f"Ganho do eco: {a1}")


if __name__ == "__main__":
    main()
