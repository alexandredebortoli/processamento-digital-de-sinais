import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

fs = 44100  # Hz
fc = 1000  # Hz
fb = 600  # Hz


def mf_peak(fc, G, fs, fb):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    k = H0 / 2
    d = -np.cos((2 * np.pi * fc) / fs)

    # Prewarp and handle boost/cut case
    if G >= 0:
        aB = (np.tan(np.pi * fb / fs) - 1) / (np.tan(np.pi * fb / fs) + 1)
    else:
        aB = (np.tan(np.pi * fb / fs) - V0) / (np.tan(np.pi * fb / fs) + V0)

    # Derived from algebraic H(z)
    b0 = 1 + k * (1 + aB)
    b1 = d - d * aB
    b2 = -(k * (aB + 1) + aB)
    a0 = 1
    a1 = d - d * aB
    a2 = -aB

    # Numerator [a0, a1, a2], Denominator [1, b1, b2]
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    return b, a


b_mf_peak_boost, a_mf_peak_boost = mf_peak(fc=fc, G=10, fs=fs, fb=fb)
b_mf_peak_cut, a_mf_peak_cut = mf_peak(fc=fc, G=-10, fs=fs, fb=fb)


audio = np.fromfile("input/sweep_20_3k4.pcm", dtype=np.int16)
y_boost = lfilter(b_mf_peak_boost, a_mf_peak_boost, audio)
y_cut = lfilter(b_mf_peak_cut, a_mf_peak_cut, audio)

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
plt.plot(t, y_boost, label="HF Shelving Boost", color="green")
plt.title("Sinal com Boost (G=10 dB)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Cut signal
plt.subplot(3, 1, 3)
plt.plot(t, y_cut, label="HF Shelving Cut", color="red")
plt.title("Sinal com Cut (G=-10 dB)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
