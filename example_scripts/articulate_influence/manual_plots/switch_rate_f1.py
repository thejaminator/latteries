import matplotlib.pyplot as plt
import numpy as np

# Define the range of switch rates from 0 to 1.
s = np.linspace(0, 1, 500)

# Compute the F1 score for each switch rate.
F1 = 2 * s / (1 + s)

# Create the plot.
plt.figure(figsize=(8, 5))
plt.plot(s, F1, label=r"$F1(s) = \frac{2s}{1+s}$", color="blue", linewidth=2)
plt.xlabel("Switch Rate (s)", fontsize=14)
plt.ylabel("Baseline F1 Score", fontsize=14)
plt.title("Baseline F1 vs. Switch Rate", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.ylim(0, 1.05)  # Ensuring the y-axis goes a bit above 1 for clarity

# Display the plot.
plt.show()
