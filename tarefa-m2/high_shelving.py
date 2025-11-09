# 1 - Em Python projete os filtros shelving LF, Peak e Shelving HF
# e plote suas respectivas frequências.

# 1-a H(Z) = 1 + H0/2 * [1- A(Z)]

# A(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y(Z)/X(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y1[n] = aB * X[n] + X[n-1] - aB * Y[n-1]

# Shelving HF para G=10dB e isso vai gerar um burst
# V0 = 10^(G/20) = 10^(10/20) = 3.162
# H0 = V0 - 1 = 3.162 - 1 = 2.162
# fc = 1000 Hz
# fs = 44100 Hz

# aB = [tan(pi * fc / fs) - 1] / [tan(pi * fc / fs) + 1] = [tan(pi * 1000 / 44100) - 1] / [tan(pi * 1000 / 44100) + 1] = 0.0086
# wc = 2 * pi * fc = 2 * pi * 1000 = 6283.185

# Y1[n] = aB * X[n] + X[n-1] - aB * Y[n-1]
# Y1[n] = 0.0086 * X[n] + X[n-1] - 0.0086 * Y[n-1]
# H(Z) = 1 + H0/2 * [1 - (Z^(-1) + aB) / (1 + aB * Z^(-1))]
# H(Z) = 1 + k * [1 - (Z^(-1) + aB) / (1 + aB * Z^(-1))]
# H(Z) = 1 + k * [[(1 + aB*Z^(-1)) - (Z^(-1) + aB)] / (1 + aB * Z^(-1))]
# H(Z) = 1 + k * [[1 + aB*Z^(-1)) - Z^(-1) - aB] / (1 + aB * Z^(-1))]
# H(Z) = 1 + k * [[1 -ab + (aB - 1)Z^(-1)] / (1 + aB * Z^(-1))]
# H(Z) = 1 + [[k(1 -ab) + k(aB - 1)Z^(-1)] / (1 + aB * Z^(-1))]
# H(Z) = [(1 + aB * Z^(-1)) + k(1 -ab) + k(aB - 1)Z^(-1)] / (1 + aB * Z^(-1))
# H(Z) = [(1 + k(1 -ab) + aBZ^(-1) + k(aB - 1)Z^(-1)] / (1 + aB * Z^(-1))
# H(Z) = [(1 + k(1 -ab) + [aB + k(aB - 1)]Z^(-1)] / (1 + aB * Z^(-1))
# H(Z) = a0 + a1Z^(-1)] / (b0 + b1* Z^(-1))

# a0 = 1 + k(1 -ab)
# a1 = aB + k(aB - 1)
# b0 = 1
# b1 = aB

# y[n] = a0*x[n] + a1*x[n-1] - b1*y[n-1]

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

fs = 44100  # Hz
fc = 1000  # Hz


def shelving_high(fc, G, fs):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    k = H0 / 2

    # Prewarp and handle boost/cut case
    if G >= 0:
        aB = (np.tan(np.pi * fc / fs) - 1) / (np.tan(np.pi * fc / fs) + 1)
    else:
        aB = (V0 * np.tan(np.pi * fc / fs) - 1) / (V0 * np.tan(np.pi * fc / fs) + 1)

    # Derived from algebraic H(z)
    a0 = 1 + k * (1 - aB)
    a1 = aB + k * (aB - 1)
    b0 = 1
    b1 = aB

    # Numerator [a0, a1], Denominator [b0, b1]
    b = np.array([a0, a1])
    a = np.array([b0, b1])

    return b, a


def plot_filter(b, a, fs, label):
    w, h = freqz(b, a, fs=fs)
    plt.plot(w, 20 * np.log10(abs(h)), label=label)


b_lf_boost, a_lf_boost = shelving_high(fc=fc, G=10, fs=fs)
b_lf_cut, a_lf_cut = shelving_high(fc=fc, G=-10, fs=fs)

plt.figure(figsize=(10, 6))
plot_filter(b_lf_boost, a_lf_boost, fs, "HF Shelving Boost")
plot_filter(b_lf_cut, a_lf_cut, fs, "HF Shelving Cut")

plt.axvline(fc, color="r", linestyle="--", label=f"Frequência de corte (fc={fc} Hz)")

plt.title("Shelving HF")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Ganho (dB)")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.show()
