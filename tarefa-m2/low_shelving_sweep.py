import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

fs = 44100  # Hz
fc = 1000  # Hz


def shelving_low(fc, G, fs):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    k = H0 / 2

    # Prewarp and handle boost/cut case
    if G >= 0:
        aB = (np.tan(np.pi * fc / fs) - 1) / (np.tan(np.pi * fc / fs) + 1)
    else:
        aB = (np.tan(np.pi * fc / fs) - V0) / (np.tan(np.pi * fc / fs) + V0)

    # Derived from algebraic H(z)
    a0 = 1 + k * (1 + aB)
    a1 = aB + k * (1 + aB)
    b1 = aB

    # Numerator [a0, a1], Denominator [1, b1]
    b = np.array([a0, a1])
    a = np.array([1, b1])

    return b, a


b_lf_boost, a_lf_boost = shelving_low(fc=fc, G=10, fs=fs)
b_lf_cut, a_lf_cut = shelving_low(fc=fc, G=-10, fs=fs)


audio = np.fromfile("input/sweep_20_3k4.pcm", dtype=np.int16)
y_boost = lfilter(b_lf_boost, a_lf_boost, audio)
y_cut = lfilter(b_lf_cut, a_lf_cut, audio)

# Create time axis
t = np.arange(len(audio)) / fs

# Plot in separate subplots
plt.figure(figsize=(12, 10))

# Input signal
plt.subplot(3, 1, 1)
plt.plot(t, audio, label="Input")
plt.title("Sinal Original")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Boost signal
plt.subplot(3, 1, 2)
plt.plot(t, y_boost, label="Boost", color="green")
plt.title("Sinal com Boost (G=10 dB)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Cut signal
plt.subplot(3, 1, 3)
plt.plot(t, y_cut, label="Cut", color="red")
plt.title("Sinal com Cut (G=-10 dB)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
