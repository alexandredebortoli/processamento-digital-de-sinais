import numpy as np
import matplotlib.pyplot as plt


def main():
    FS = 8000
    t1 = 1.0 * 10**-3
    t2 = 1.5 * 10**-3

    # Exemplo de delay
    n1 = t1 * FS
    n2 = t2 * FS

    # Definição dos ganhos
    a0 = 0.5
    a1 = 0.3
    a2 = 0.2
    tama_delay = n2
    vetor_delay = np.zeros(tama_delay, 1)

    # Definindo a entrada
    entrada = np.zeros(2 * tama_delay, 1)

    entrada[1, 1] = 1  # Definindo o umpulso unitário

    tama_loop = len(entrada)
    vet_saida = np.zeros(tama_loop, 1)

    for j in range(1, tama_loop):
        input = entrada[j, 1]
        vetor_delay[1, 1] = input
        vet_saida[j, 1] = a0 * input + a1 * vetor_delay[1, 1] + a2 * vetor_delay[1, 2]
        vetor_delay[1, 2] = vetor_delay[1, 1]
        vetor_delay[1, 1] = input

    plt.plot(vet_saida)
    plt.show()


if __name__ == "__main__":
    main()
