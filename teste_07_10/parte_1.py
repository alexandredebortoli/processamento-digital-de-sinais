import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================================
# Definição do sistema H(z)
# Numerador e denominador em z^-1
# H(z) = (0.7294 - 2.1883 z^-1 + 2.1883 z^-2 - 0.7294 z^-3) /
#        (1 - 2.3741 z^-1 + 1.9294 z^-2 - 0.5321 z^-3)
# =========================================

# Coeficientes do numerador (b) e denominador (a)
b = [0.7294, -2.1883, 2.1883, -0.7294]
a = [1.0, -2.3741, 1.9294, -0.5321]

# Frequência de amostragem
fs = 8000

# =========================================
# a) Polos e zeros
# =========================================
zeros = np.roots(b)
poles = np.roots(a)

print("Zeros:", zeros)
print("Polos:", poles)

# =========================================
# b) Estabilidade (polos dentro do círculo unitário)
# =========================================
stable = np.all(np.abs(poles) < 1)
print("Sistema é estável?", stable)

# Gráfico do plano-z (polos e zeros)
plt.figure(figsize=(5, 5))
plt.axhline(0, color="black")
plt.axvline(0, color="black")
circle = plt.Circle((0, 0), 1, color="blue", fill=False, linestyle="dashed")
plt.gca().add_artist(circle)
plt.scatter(
    np.real(zeros),
    np.imag(zeros),
    marker="o",
    facecolors="none",
    edgecolors="g",
    label="Zeros",
)
plt.scatter(np.real(poles), np.imag(poles), marker="x", color="r", label="Polos")
plt.title("Polos e Zeros no Plano-z")
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.legend()
plt.grid(True)
plt.axis("equal")

# =========================================
# c) Resposta em frequência
# =========================================
w, h = signal.freqz(b, a, worN=512, fs=fs)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title("Resposta em Frequência")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.axvline(100, color="r", linestyle="--")
plt.axvline(1000, color="r", linestyle="--")
mag_100 = 20 * np.log10(np.abs(np.interp(100, w, np.abs(h))))
mag_1000 = 20 * np.log10(np.abs(np.interp(1000, w, np.abs(h))))
plt.text(100, mag_100, f"{mag_100:.2f} dB", color="r", va="bottom", ha="right")
plt.text(1000, mag_1000, f"{mag_1000:.2f} dB", color="r", va="bottom", ha="right")

plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h))
plt.ylabel("Fase (rad)")
plt.xlabel("Frequência (Hz)")
plt.grid(True)

# =========================================
# d) Equação a diferenças
# Forma: y[n] - 2.3741 y[n-1] + 1.9294 y[n-2] - 0.5321 y[n-3] =
#        0.7294 x[n] - 2.1883 x[n-1] + 2.1883 x[n-2] - 0.7294 x[n-3]
# =========================================

print("\nEquação a diferenças (em n):")
print(
    "y[n] - 2.3741 y[n-1] + 1.9294 y[n-2] - 0.5321 y[n-3] = "
    "0.7294 x[n] - 2.1883 x[n-1] + 2.1883 x[n-2] - 0.7294 x[n-3]"
)

plt.show()
