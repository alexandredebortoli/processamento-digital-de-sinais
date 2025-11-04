# 1 - Em Python projete os filtros shelving LF, Peak e Shelving HF
# e plote suas respectivas frequências.

# 1-a H(Z) = 1 + H0/2 * [1+ A(Z)]

# A(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y(Z)/X(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y[n] = aB * X[n] + X[n-1] - aB * Y[n-1]

# Shelving LF para G=10dB e isso vai gerar um burst
# V0 = 10^(G/20) = 10^(10/20) = 3.162
# H0 = V0 - 1 = 3.162 - 1 = 2.162
# fc = 1000 Hz
# fs = 44100 Hz

# aB = [tan(pi * fc / fs) - 1] / [tan(pi * fc / fs) + 1] = [tan(pi * 1000 / 44100) - 1] / [tan(pi * 1000 / 44100) + 1] = 0.0086
# wc = 2 * pi * fc = 2 * pi * 1000 = 6283.185

# Y[n] = aB * X[n] + X[n-1] - aB * Y[n-1]
# Y[n] = 0.0086 * X[n] + X[n-1] - 0.0086 * Y[n-1]

# 1-b H(Z) = 1 + H0/2 * [1+ A(Z)]
# A(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y(Z)/X(Z) = (Z^(-1) + aB) / (1 + aB * Z^(-1))
# Y[n] = aB * X[n] + X[n-1] - aB * Y[n-1]

# Shelving HF para G=10dB e isso vai gerar um burst
# V0 = 10^(G/20) = 10^(10/20) = 3.162
# H0 = V0 - 1 = 3.162 - 1 = 2.162


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

fs = 44100  # Hz


# ============================================================
# 1. LF Shelving (Low-frequency boost)
# ============================================================
def shelving_low(fc, G, fs):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    aB = (np.tan(np.pi * fc / fs) - 1) / (np.tan(np.pi * fc / fs) + 1)
    b0 = 1 + H0 / 2 * (1 + aB)
    b1 = H0 / 2 * (1 + aB) - aB
    a1 = -aB
    return np.array([b0, b1]), np.array([1, a1])


# ============================================================
# 2. Peak Filter
# ============================================================
def peak_filter(fc, G, Q, fs):
    V0 = 10 ** (G / 20)
    K = np.tan(np.pi * fc / fs)
    norm = 1 / (1 + K / Q + K**2)
    b0 = (1 + V0 * K / Q + K**2) * norm
    b1 = 2 * (K**2 - 1) * norm
    b2 = (1 - V0 * K / Q + K**2) * norm
    a0 = 1
    a1 = b1
    a2 = (1 - K / Q + K**2) * norm
    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


# ============================================================
# 3. HF Shelving (High-frequency boost)
# ============================================================
def shelving_high(fc, G, fs):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    aB = (np.tan(np.pi * fc / fs) - 1) / (np.tan(np.pi * fc / fs) + 1)
    b0 = 1 + H0 / 2 * (1 - aB)
    b1 = H0 / 2 * (aB - 1) + aB
    a1 = -aB
    return np.array([b0, b1]), np.array([1, a1])


# ============================================================
# 4. Plotagem das respostas em frequência
# ============================================================
def plot_filter(b, a, fs, label):
    w, h = freqz(b, a, fs=fs)
    plt.plot(w, 20 * np.log10(abs(h)), label=label)


# ============================================================
# Criando filtros
# ============================================================
b_lf, a_lf = shelving_low(fc=1000, G=10, fs=fs)
b_mf1, a_mf1 = peak_filter(fc=3000, G=5, Q=1, fs=fs)
b_mf2, a_mf2 = peak_filter(fc=8000, G=-3, Q=1, fs=fs)
b_hf, a_hf = shelving_high(fc=12000, G=8, fs=fs)

# ============================================================
# Plot individual
# ============================================================
plt.figure(figsize=(10, 6))
plot_filter(b_lf, a_lf, fs, "LF Shelving")
plot_filter(b_mf1, a_mf1, fs, "MF Peak 1")
plot_filter(b_mf2, a_mf2, fs, "MF Peak 2")
plot_filter(b_hf, a_hf, fs, "HF Shelving")

plt.title("Equalizador – Respostas Individuais")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Ganho (dB)")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.show()
