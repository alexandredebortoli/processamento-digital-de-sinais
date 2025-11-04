import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

fs = 48000
fc = 500
G = 10
A = 10 ** (G / 40)
w0 = 2 * np.pi * fc / fs
alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 1) - 2)

b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

b = np.array([b0, b1, b2]) / a0
a = np.array([1, a1 / a0, a2 / a0])

w, h = signal.freqz(b, a, worN=2048)
f = w * fs / (2 * np.pi)

plt.figure(figsize=(8, 4))
plt.semilogx(f, 20 * np.log10(abs(h)))
plt.title("SHELVING LF")
plt.xlabel("FREQUENCY [Hz]")
plt.ylabel("GAIN [dB]")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.axvline(fc, color="red", linestyle="--", label=f"fc = {fc} Hz")
plt.axhline(G, color="green", linestyle=":", label=f"G = {G} dB")
plt.legend()
plt.show()
