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


def main():
    x = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    K = 10

    coefficient_vector = np.array([1 / K] * K)

    sample_vector = np.zeros(K)

    y = np.zeros(len(x))

    for n in range(len(x)):
        sample_vector[1:] = sample_vector[:-1]
        sample_vector[0] = x[n]

        y[n] = moving_average_processing(sample_vector, coefficient_vector)

    # Print results with simple but pretty formatting
    print("\n" + "─" * 50)
    print("📊 MOVING AVERAGE RESULTS")
    print("─" * 50)
    print(f"🔧 K = {K}")
    print(f"📥 Input:  [{', '.join([f'{val:.0f}' for val in x])}]")
    print(f"📤 Output: [{', '.join([f'{val:.2f}' for val in y])}]")
    print("─" * 50)

    # Plot the impulse response
    n = np.arange(len(x))

    plt.figure(figsize=(10, 6))

    # Plot input
    plt.subplot(2, 1, 1)
    plt.stem(n, x, "b-", label="Input x[n]")
    plt.title("Input Signal (Impulse)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot output
    plt.subplot(2, 1, 2)
    plt.stem(n, y, "r-", label="Output y[n]")
    plt.title("Output Signal (Derivative)")
    plt.xlabel("Sample n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
