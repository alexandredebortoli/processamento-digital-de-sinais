import numpy as np
import matplotlib.pyplot as plt


def moving_average_processing(sample_vector, coefficient_vector):
    """
    Simple moving average system: y[n] = a0* x[n] + a1* x[n-1] + a2* x[n-2] + a3* x[n-3]
    """
    output = 0
    for n in range(0, len(sample_vector)):
        output += coefficient_vector[n] * sample_vector[n]
    return output


def process_moving_average(x, K):
    """Process moving average for a specific K value"""
    coefficient_vector = np.array([1 / K] * K)
    sample_vector = np.zeros(K)
    y = np.zeros(len(x))

    for n in range(len(x)):
        sample_vector[1:] = sample_vector[:-1]
        sample_vector[0] = x[n]
        y[n] = moving_average_processing(sample_vector, coefficient_vector)

    return y


def plot_and_save(x, y, K, output_dir):
    """Create plot and save it to output directory"""
    n = np.arange(len(x))

    plt.figure(figsize=(12, 8))

    # Plot output
    plt.subplot(2, 1, 2)
    plt.stem(n, y, "r-", label="Output y[n]", markerfmt="o", linefmt="r-")
    plt.title(f"Output Signal (Moving Average) - K={K}")
    plt.xlabel("Sample n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save plot
    filename = f"moving_average_k{K}.png"
    filepath = f"{output_dir}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"💾 Saved: {filename}")
    plt.close()


def main():
    # Read PCM file
    try:
        x = np.fromfile("sweep_20_3k4.pcm", dtype=np.int16)
        print(f"📁 Loaded PCM file: {len(x)} samples")
    except FileNotFoundError:
        print("❌ Error: sweep_20_3k4.pcm not found!")
        return

    # Create output directory
    import os

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")

    # Test different K values
    K_values = [4, 8, 16, 32]

    print("\n" + "─" * 50)
    print("📊 MOVING AVERAGE ANALYSIS")
    print("─" * 50)

    for K in K_values:
        print(f"\n🔧 Processing K = {K}")
        y = process_moving_average(x, K)

        # Print summary statistics
        print(f"   • Max output: {np.max(y):.2f}")
        print(f"   • Min output: {np.min(y):.2f}")
        print(f"   • Output range: {np.max(y) - np.min(y):.2f}")

        # Plot and save
        plot_and_save(x, y, K, output_dir)

    print("\n" + "─" * 50)
    print("✅ All plots saved to 'output' folder!")
    print("─" * 50)


if __name__ == "__main__":
    main()
