import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    plt.close("all")

    # Arquivo de áudio
    input_file = "sweep_20_3k4.pcm"
    if not os.path.exists(input_file):
        print(f"Arquivo '{input_file}' não encontrado!")
        return

    # Lê o arquivo de áudio
    with open(input_file, "rb") as fid:
        entrada = np.frombuffer(fid.read(), dtype=np.int16)

    print(f"Processando {len(entrada)} amostras...")

    # Parâmetros do eco
    Fs = 8000  # taxa de amostragem
    Ts = 1 / Fs  # período de amostragem (125us)
    D = 300e-3  # atraso de 300ms (bem perceptível)
    delay_samples = int(D * Fs)  # 2400 amostras

    # Ganhos
    a0 = 0.8
    a1 = 0.6

    # Buffer de delay
    buffer_delay = np.zeros(delay_samples)

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

    # Normaliza para evitar clipping
    saida = np.clip(saida, -32767, 32767).astype(np.int16)

    # Mostra os resultados
    plt.figure(figsize=(12, 8))

    # Sinal original
    plt.subplot(3, 1, 1)
    plt.plot(entrada[:8000])
    plt.title("Sinal original")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Sinal com eco
    plt.subplot(3, 1, 2)
    plt.plot(saida[:8000])
    plt.title("Sinal com eco")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Comparação
    plt.subplot(3, 1, 3)
    plt.plot(entrada[:8000], alpha=0.7, label="Original")
    plt.plot(saida[:8000], alpha=0.7, label="Com eco")
    plt.title("Comparação")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Salva o gráfico
    plt.savefig("eco_sweep.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo!")

    plt.show()

    # Salva o arquivo com eco
    with open("sweep_com_eco.pcm", "wb") as fid:
        saida.tofile(fid)
    print("Arquivo com eco salvo!")

    # Mostra informações
    print(f"Taxa de amostragem: {Fs} Hz")
    print(f"Período de amostragem: {Ts*1e6:.1f} μs")
    print(f"Atraso: {D*1000:.1f} ms")
    print(f"Atraso em amostras: {delay_samples}")
    print(f"Ganho original: {a0}")
    print(f"Ganho do eco: {a1}")


if __name__ == "__main__":
    main()
