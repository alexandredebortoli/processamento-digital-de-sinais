import numpy as np
import matplotlib.pyplot as plt

# Caminhos dos arquivos
input_file = "sweep_novo.pcm"
output_file = "sweep_saida.pcm"

# Leitura do arquivo PCM (16-bit little endian)
with open(input_file, "rb") as f:
    pcm_bytes = f.read()
    x_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

# Normalização para float em [-1, 1]
x = x_int16.astype(np.float32) / 32768.0

print(f"Tamanho do sinal de entrada: {len(x)} samples")
print(f"Duração: {len(x)/8000:.2f} segundos")

# Implementação da equação a diferenças
# y[n] = 0.7294 x[n] - 2.1883 x[n-1] + 2.1883 x[n-2] - 0.7294 x[n-3]
#        + 2.3741 y[n-1] - 1.9294 y[n-2] + 0.5321 y[n-3]

# Inicializar o vetor de saída
y = np.zeros_like(x)

# Implementar a equação a diferenças
for n in range(len(x)):
    # Termos de entrada
    term_x0 = 0.7294 * x[n] if n >= 0 else 0
    term_x1 = -2.1883 * x[n - 1] if n >= 1 else 0
    term_x2 = 2.1883 * x[n - 2] if n >= 2 else 0
    term_x3 = -0.7294 * x[n - 3] if n >= 3 else 0

    # Termos de saída (feedback)
    term_y1 = 2.3741 * y[n - 1] if n >= 1 else 0
    term_y2 = -1.9294 * y[n - 2] if n >= 2 else 0
    term_y3 = 0.5321 * y[n - 3] if n >= 3 else 0

    # Calcular y[n]
    y[n] = term_x0 + term_x1 + term_x2 + term_x3 + term_y1 + term_y2 + term_y3

# Re-normalização para int16
y_int16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)

# Salvando a saída
with open(output_file, "wb") as f:
    f.write(y_int16.tobytes())

print(f"Tamanho do sinal de saída: {len(y_int16)} samples")
print(f"Arquivo de saída salvo como {output_file}")

# Plotagem dos sinais completos
plt.figure(figsize=(15, 10))

# Plot do sinal de entrada completo
plt.subplot(3, 1, 1)
plt.plot(x, label="Sinal de Entrada")
plt.title(f"Sinal de Entrada Completo ({len(x)} samples)")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Plot do sinal de saída completo
plt.subplot(3, 1, 2)
plt.plot(y, label="Sinal de Saída")
plt.title(f"Sinal de Saída Completo ({len(y)} samples)")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Análise espectral completa
fs = 8000  # Frequência de amostragem
plt.subplot(3, 1, 3)

# FFT dos sinais completos
X_fft = np.fft.fft(x)
Y_fft = np.fft.fft(y)
freqs = np.fft.fftfreq(len(x), 1 / fs)

# Plotar apenas frequências positivas
pos_freqs = freqs[: len(freqs) // 2]
plt.plot(
    pos_freqs,
    20 * np.log10(np.abs(X_fft[: len(X_fft) // 2])),
    label="Entrada (dB)",
    alpha=0.7,
)
plt.plot(
    pos_freqs,
    20 * np.log10(np.abs(Y_fft[: len(Y_fft) // 2])),
    label="Saída (dB)",
    alpha=0.7,
)
plt.title("Espectro de Frequência Completo")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.xlim(0, fs / 2)

plt.tight_layout()
plt.show()

# Análise estatística dos sinais
print(f"\n=== ANÁLISE ESTATÍSTICA ===")
print(f"Entrada - RMS: {np.sqrt(np.mean(x**2)):.6f}")
print(f"Saída - RMS: {np.sqrt(np.mean(y**2)):.6f}")
print(f"Entrada - Amplitude máxima: {np.max(np.abs(x)):.6f}")
print(f"Saída - Amplitude máxima: {np.max(np.abs(y)):.6f}")

# Encontrar frequências com energia significativa
X_mag_db = 20 * np.log10(np.abs(X_fft[: len(X_fft) // 2]))
Y_mag_db = 20 * np.log10(np.abs(Y_fft[: len(Y_fft) // 2]))

threshold = -40
X_significant = np.where(X_mag_db > threshold)[0]
Y_significant = np.where(Y_mag_db > threshold)[0]

if len(X_significant) > 0:
    freq_range_input = pos_freqs[X_significant]
    print(
        f"Faixa de frequências no sinal de entrada: {freq_range_input[0]:.1f} - {freq_range_input[-1]:.1f} Hz"
    )

if len(Y_significant) > 0:
    freq_range_output = pos_freqs[Y_significant]
    print(
        f"Faixa de frequências no sinal de saída: {freq_range_output[0]:.1f} - {freq_range_output[-1]:.1f} Hz"
    )

print(f"\nProcessamento concluído!")
