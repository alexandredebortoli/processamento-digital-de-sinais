import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, freqz, lfilter

fs = 44100  # Hz
fb_peak_1 = 1000  # Hz
fb_peak_2 = 300  # Hz

fc_low = 1000  # Hz
fc_peak_1 = 3000  # Hz
fc_peak_2 = 6000  # Hz
fc_high = 10000  # Hz


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


b_lf, a_lf = shelving_low(fc=fc_low, G=10, fs=fs)
b_mf_peak_1, a_mf_peak_1 = mf_peak(fc=fc_peak_1, G=10, fs=fs, fb=fb_peak_1)
b_mf_peak_2, a_mf_peak_2 = mf_peak(fc=fc_peak_2, G=10, fs=fs, fb=fb_peak_2)
b_hf, a_hf = shelving_high(fc=fc_high, G=10, fs=fs)

b_combined = convolve(convolve(convolve(b_lf, b_mf_peak_1), b_mf_peak_2), b_hf)
a_combined = convolve(convolve(convolve(a_lf, a_mf_peak_1), a_mf_peak_2), a_hf)


input_signal = np.fromfile("input/sweep_20_3k4.pcm", dtype=np.int16).astype(float)
output_signal = lfilter(b_combined, a_combined, input_signal)

max_val = np.max(np.abs(output_signal))
if max_val > 0:
    output_signal = output_signal * (32767 / max_val)

output_signal = np.clip(output_signal, -32768, 32767).astype(np.int16)


# Create time axis
t = np.arange(len(input_signal))

# Plot in separate subplots
plt.figure(figsize=(12, 10))

# Input signal
plt.subplot(2, 1, 1)
plt.plot(t, input_signal, label="Input")
plt.title("Sinal Original")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Boost signal
plt.subplot(2, 1, 2)
plt.plot(t, output_signal, label="Output", color="green")
plt.title(
    "Sinal com Equalizer (G=10 dB) para 1000 Hz (LF), 3000 Hz (MF), 6000 Hz (MF) e 10000 Hz (HF)"
)
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# clip and save
output_signal.tofile("output/equalizer.pcm")
