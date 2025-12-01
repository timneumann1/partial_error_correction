import matplotlib.pyplot as plt
import numpy as np

# RANDOM CIRCUITS
rand_plain_base = np.array([0.3484000000000001, 0.273025, 0.34764999999999996, 0.34702000000000005, 0.34799500000000017, 0.2718550000000001, 0.3460750000000002, 0.3484000000000001, 0.34697, 0.27289])
rand_ft_base = np.array([0.655315, 0.5645700000000002, 0.656535, 0.6559600000000001, 0.658255, 0.5653349999999998, 0.65694, 0.6577200000000002, 0.65684, 0.5644550000000002])
rand_improve_base = rand_ft_base - rand_plain_base

rand_plain_new = np.array([0.3496800000000002, 0.27246000000000004, 0.34641, 0.3469000000000002, 0.3450249999999999, 0.27343500000000015, 0.3472850000000001, 0.3464399999999999, 0.3462950000000002, 0.27173000000000014])
rand_ft_new = np.array([0.6550949999999999, 0.576115, 0.654355, 0.654315, 0.6524700000000001, 0.5759800000000003, 0.6530949999999999, 0.6553650000000001, 0.6540599999999999, 0.577885])
rand_improve_new = rand_ft_new - rand_plain_new

x = np.arange(1, 11)

plt.figure(figsize=(10,6))

# Plain
plt.plot(x, rand_plain_base, 'o-', color='blue', label='Plain')

# Baseline
plt.plot(x, rand_ft_base, 's-', color='orange', label='Baseline')
plt.plot(x, rand_ft_new, 's-', color='green', label='Lookahead')

plt.xlabel('Circuit Index', size = 14)
plt.ylabel('Fidelity', size = 14)
plt.title('Comparison of Plain, Baseline, and Lookahead Model', size = 14)
plt.suptitle("Quantum Phase Estimation on T operator in low-qubit regime", size=18)
plt.ylim(.20, .95)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/qpe_plot.png", dpi=300)