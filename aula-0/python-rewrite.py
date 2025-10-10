import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    plt.close("all")

    # Arquivo que vamos processar
    input_file = "sweep_20_3k4.pcm"
    if not os.path.exists(input_file):
        print(f"Arquivo '{input_file}' não encontrado!")
        return

    # Lê o arquivo de áudio
    try:
        with open(input_file, "rb") as fid:
            s = np.frombuffer(fid.read(), dtype=np.int16)
    except Exception as e:
        print(f"Erro ao ler: {e}")
        return

    print(f"Processando {len(s)} amostras...")

    # Prepara o gráfico
    plt.figure(figsize=(10, 8))

    # Mostra o sinal original
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.grid(True)
    plt.title("Sinal de entrada")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")

    # Aplica o ganho
    ganho = 2
    resultado = np.zeros(len(s), dtype=np.int16)

    for i in range(len(s)):
        resultado[i] = ganho * s[i]

    # Mostra o resultado
    plt.subplot(2, 1, 2)
    plt.plot(resultado, "r")
    plt.title("Sinal processado")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()

    # Salva o gráfico
    plt.savefig("grafico_sinal.png", dpi=300, bbox_inches="tight")
    print("Gráfico salvo!")

    plt.show()

    # Salva o arquivo processado
    with open("sinal_saida.pcm", "wb") as fid:
        resultado.tofile(fid)
    print("Arquivo salvo!")


if __name__ == "__main__":
    main()
