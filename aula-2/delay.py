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

    # Parâmetros do delay
    Fs = 8000  # taxa de amostragem
    delay_time = 200e-3  # 200ms de delay
    delay_samples = int(delay_time * Fs)

    # Ganhos para o efeito
    ganho_original = 0.7
    ganho_delay = 0.5

    # Buffer de delay
    buffer_delay = np.zeros(delay_samples)

    # Saída processada
    saida = np.zeros(len(entrada))

    # Aplica o delay
    for i in range(len(entrada)):
        # Entrada atual
        x = entrada[i]

        # Saída = sinal original + sinal atrasado
        y = ganho_original * x + ganho_delay * buffer_delay[-1]

        # Atualiza o buffer de delay
        buffer_delay[1:] = buffer_delay[:-1]
        buffer_delay[0] = x

        saida[i] = y

    # Normaliza para evitar clipping
    saida = np.clip(saida, -32767, 32767).astype(np.int16)

    # Mostra uma parte do sinal
    plt.figure(figsize=(12, 8))

    # Sinal original
    plt.subplot(3, 1, 1)
    plt.plot(entrada[:8000])
    plt.title("Sinal original")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Sinal com delay
    plt.subplot(3, 1, 2)
    plt.plot(saida[:8000])
    plt.title("Sinal com delay")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Comparação
    plt.subplot(3, 1, 3)
    plt.plot(entrada[:8000], alpha=0.7, label="Original")
    plt.plot(saida[:8000], alpha=0.7, label="Com delay")
    plt.title("Comparação")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Salva o gráfico
    plt.savefig("delay_audio.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo!")

    plt.show()

    # Salva o arquivo com delay
    with open("sweep_com_delay.pcm", "wb") as fid:
        saida.tofile(fid)
    print("Arquivo com delay salvo!")


if __name__ == "__main__":
    main()
