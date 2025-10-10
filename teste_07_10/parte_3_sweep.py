import numpy as np
import matplotlib.pyplot as plt

# Caminhos dos arquivos
input_file = "sweep_20_3k4.pcm"
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

# --- Geração de senos e cálculo de atenuação ---
print(f"\n=== ANÁLISE DE SENOS ===")

# Parâmetros
fs = 8000  # Frequência de amostragem
duration = 1.0  # Duração em segundos
t = np.arange(0, duration, 1 / fs)

# Geração dos senos
f1 = 100  # Hz
f2 = 1000  # Hz

# Seno de 100 Hz
x1 = np.sin(2 * np.pi * f1 * t)
y1 = np.zeros_like(x1)

# Seno de 1 kHz
x2 = np.sin(2 * np.pi * f2 * t)
y2 = np.zeros_like(x2)


# Aplicar o filtro H(z) aos senos usando a mesma equação a diferenças
def apply_filter(input_signal):
    output = np.zeros_like(input_signal)
    for n in range(len(input_signal)):
        # Termos de entrada
        term_x0 = 0.7294 * input_signal[n] if n >= 0 else 0
        term_x1 = -2.1883 * input_signal[n - 1] if n >= 1 else 0
        term_x2 = 2.1883 * input_signal[n - 2] if n >= 2 else 0
        term_x3 = -0.7294 * input_signal[n - 3] if n >= 3 else 0

        # Termos de saída (feedback)
        term_y1 = 2.3741 * output[n - 1] if n >= 1 else 0
        term_y2 = -1.9294 * output[n - 2] if n >= 2 else 0
        term_y3 = 0.5321 * output[n - 3] if n >= 3 else 0

        # Calcular y[n]
        output[n] = term_x0 + term_x1 + term_x2 + term_x3 + term_y1 + term_y2 + term_y3

    return output


# Aplicar filtro aos senos
y1 = apply_filter(x1)
y2 = apply_filter(x2)


# Função para calcular pico (amplitude máxima)
def peak(signal):
    return np.max(np.abs(signal))


# Calcular atenuação em dB usando picos
peak_in_100 = peak(x1)
peak_out_100 = peak(y1)
print(f"Peak in 100 Hz: {peak_in_100}")
print(f"Peak out 100 Hz: {peak_out_100}")
ATdB_100 = 20 * np.log10(peak_out_100 / peak_in_100)

peak_in_1000 = peak(x2)
peak_out_1000 = peak(y2)
print(f"Peak in 1000 Hz: {peak_in_1000}")
print(f"Peak out 1000 Hz: {peak_out_1000}")
ATdB_1000 = 20 * np.log10(peak_out_1000 / peak_in_1000)

print(f"Atenuação para {f1} Hz: {ATdB_100:.2f} dB")
print(f"Atenuação para {f2} Hz: {ATdB_1000:.2f} dB")

# Plot dos sinais de entrada e saída
plt.figure(figsize=(15, 10))

# Plot para 100 Hz
plt.subplot(3, 2, 1)
N_plot_100 = int(4 * fs / f1)  # 4 períodos
plt.plot(t[:N_plot_100], x1[:N_plot_100], label=f"Entrada {f1} Hz", linewidth=2)
plt.plot(t[:N_plot_100], y1[:N_plot_100], label=f"Saída {f1} Hz", linewidth=2)
plt.title(f"{f1} Hz - Entrada e Saída")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Plot para 1 kHz
plt.subplot(3, 2, 2)
N_plot_1000 = int(4 * fs / f2)  # 4 períodos
plt.plot(t[:N_plot_1000], x2[:N_plot_1000], label=f"Entrada {f2} Hz", linewidth=2)
plt.plot(t[:N_plot_1000], y2[:N_plot_1000], label=f"Saída {f2} Hz", linewidth=2)
plt.title(f"{f2} Hz - Entrada e Saída")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Peak bar plots
plt.subplot(3, 2, 3)
plt.bar(
    [f"Entrada {f1} Hz", f"Saída {f1} Hz"],
    [peak_in_100, peak_out_100],
    color=["C0", "C1"],
)
plt.title(f"Peak - {f1} Hz")
plt.ylabel("Peak Amplitude")
plt.grid(axis="y")

plt.subplot(3, 2, 4)
plt.bar(
    [f"Entrada {f2} Hz", f"Saída {f2} Hz"],
    [peak_in_1000, peak_out_1000],
    color=["C0", "C1"],
)
plt.title(f"Peak - {f2} Hz")
plt.ylabel("Peak Amplitude")
plt.grid(axis="y")

# Espectros dos senos
plt.subplot(3, 2, 5)
# FFT dos senos de 100 Hz
X1_fft = np.fft.fft(x1)
Y1_fft = np.fft.fft(y1)
freqs_sine = np.fft.fftfreq(len(x1), 1 / fs)
pos_freqs_sine = freqs_sine[: len(freqs_sine) // 2]

plt.plot(
    pos_freqs_sine,
    20 * np.log10(np.abs(X1_fft[: len(X1_fft) // 2])),
    label=f"Entrada {f1} Hz",
    alpha=0.7,
)
plt.plot(
    pos_freqs_sine,
    20 * np.log10(np.abs(Y1_fft[: len(Y1_fft) // 2])),
    label=f"Saída {f1} Hz",
    alpha=0.7,
)
plt.title(f"Espectro - {f1} Hz")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.xlim(0, 2000)

plt.subplot(3, 2, 6)
# FFT dos senos de 1 kHz
X2_fft = np.fft.fft(x2)
Y2_fft = np.fft.fft(y2)

plt.plot(
    pos_freqs_sine,
    20 * np.log10(np.abs(X2_fft[: len(X2_fft) // 2])),
    label=f"Entrada {f2} Hz",
    alpha=0.7,
)
plt.plot(
    pos_freqs_sine,
    20 * np.log10(np.abs(Y2_fft[: len(Y2_fft) // 2])),
    label=f"Saída {f2} Hz",
    alpha=0.7,
)
plt.title(f"Espectro - {f2} Hz")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.xlim(0, 2000)

plt.tight_layout()
plt.show()

print(f"\nAnálise de senos concluída!")
