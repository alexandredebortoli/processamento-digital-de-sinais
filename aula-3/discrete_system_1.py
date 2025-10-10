import numpy as np
import matplotlib.pyplot as plt


def discrete_derivative_system(x):
    """
    Simple discrete derivative system: y[n] = x[n] - x[n-1]
    """
    y = np.zeros_like(x)

    # First sample: y[0] = x[0] - 0 (assuming x[-1] = 0)
    y[0] = x[0]

    # For n > 0: y[n] = x[n] - x[n-1]
    for n in range(1, len(x)):
        y[n] = x[n] - x[n - 1]

    return y


def main():
    # Simple input array - impulse at position 5
    x = np.array([0, 0, 1, 1, 1, 1, 1])

    # Apply the derivative system
    y = discrete_derivative_system(x)

    # Print results
    print("Input signal x[n]:", x)
    print("Output signal y[n]:", y)

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
