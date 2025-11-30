import matplotlib.pyplot as plt
import numpy as np

# RANDOM CIRCUITS
rand_plain_base = np.array([0.38379500000000016, 0.42658, 0.4969949999999999, 0.49848999999999993, 0.49983000000000005, 0.425505, 0.49828000000000006, 0.4984150000000001, 0.498565, 0.42484000000000016])
rand_ft_base = np.array([0.6730400000000002, 0.7232750000000001, 0.770105, 0.7694749999999999, 0.7704300000000002, 0.7245, 0.76998, 0.7703500000000002, 0.771215, 0.725515])
rand_improve_base = rand_ft_base - rand_plain_base

rand_plain_new = np.array([0.3850650000000001, 0.4251050000000001, 0.49926499999999996, 0.4985400000000002, 0.49870999999999993, 0.42597000000000007, 0.49767999999999996, 0.4970000000000002, 0.49883499999999986, 0.42452500000000004])
rand_ft_new = np.array([0.674025, 0.7082050000000001, 0.74512, 0.7483599999999999, 0.7474399999999999, 0.708765, 0.7467200000000002, 0.74644, 0.7466350000000002, 0.7065049999999999])
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
plt.title('Comparison of Plain, Baseline, and Lookahead', size = 20)
plt.ylim(.20, .95)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()