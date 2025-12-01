import numpy as np
import matplotlib.pyplot as plt


# RANDOM CIRCUITS
rand_plain_base = np.array([0.8049298965609102, 0.7335027060868324, 0.9640397283895374,
                            0.7134555345667105, 0.8377644351778425, 0.8823375540640415,
                            0.9568334177543653, 0.9833258498888917, 0.6454289437495637,
                            0.9761526930392614])
rand_ft_base = np.array([0.9275875294546794, 0.8118798071242789, 0.9913545837061701,
                         0.8538347213462685, 0.9022922636401728, 0.9474255513283898,
                         0.9787957746474455, 0.9870161273566518, 0.7411176952852186,
                         0.9972463529931315])
rand_improve_base = rand_ft_base - rand_plain_base

rand_plain_new = np.array([0.806070843433133, 0.7341227812309873, 0.9642040342062286,
                           0.714380913923934, 0.8352472074490501, 0.881539198290727,
                           0.9570772532482618, 0.9836915468227341, 0.6399900477414472,
                           0.9754304374148247])
rand_ft_new = np.array([0.9201518539425696, 0.8712021003358271, 0.9953942667705495,
                        0.8524980749159086, 0.9386737319215221, 0.9687676462240811,
                        0.9784619976028575, 0.9983421059151679, 0.7979496492336324,
                        0.9786638865712367])
rand_improve_new = rand_ft_new - rand_plain_new

# QFT CIRCUITS
qft_plain_base = np.array([0.413, 0.280935, 0.69154, 0.413615, 0.41353,
                           0.277955, 0.690765, 0.690655, 0.413585, 0.506215])
qft_ft_base = np.array([0.569815, 0.4033, 0.87681, 0.57114, 0.57223,
                        0.401255, 0.87641, 0.87549, 0.570145, 0.67305])
qft_improve_base = qft_ft_base - qft_plain_base

qft_plain_new = np.array([0.413075, 0.280185, 0.692285, 0.41294, 0.41183,
                          0.278515, 0.69331, 0.68959, 0.413755, 0.50931])
qft_ft_new = np.array([0.661805, 0.506995, 0.903, 0.660755, 0.663445,
                       0.50761, 0.902385, 0.90222, 0.66211, 0.73525])
qft_improve_new = qft_ft_new - qft_plain_new

# -----------------------------
# PLOTTING
# -----------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'width_ratios':[3,3,1]}, sharex=True)

idx = np.arange(10)

# -----------------------------
# RANDOM CIRCUITS
# -----------------------------
axes[0,0].plot(idx, rand_plain_base, "o-", label="Plain")
axes[0,0].plot(idx, rand_ft_base, "o-", label="Baseline")
axes[0,0].plot(idx, rand_ft_new, "o-", color="green", marker="o", linestyle="-", label="Lookahead")
axes[0,0].set_title("Random: Fidelities")
axes[0,0].set_ylabel("Fidelity")
axes[0,0].set_xticks([])  # remove x ticks
axes[0,0].legend()

axes[0,1].plot(idx, rand_improve_base, "o-", label="Baseline")
axes[0,1].plot(idx, rand_improve_new, "o-", color="green", marker="o", linestyle="-", label="Lookahead")
axes[0,1].set_title("Random: Per-Circuit Improvement")
axes[0,1].set_ylabel("Δ Fidelity")
axes[0,1].set_xticks([])  # remove x ticks
axes[0,1].legend()

axes[0,2].bar(["Baseline", "Lookahead"],
              [rand_improve_base.mean(), rand_improve_new.mean()],
              color=["C0", "green"])
axes[0,2].set_title("Avg. Improvement")
axes[0,2].set_ylabel("Mean Δ Fidelity")

# -----------------------------
# QFT CIRCUITS
# -----------------------------
axes[1,0].plot(idx, qft_plain_base, "o-", label="Plain")
axes[1,0].plot(idx, qft_ft_base, "o-", label="Baseline")
axes[1,0].plot(idx, qft_ft_new, "o-", color="green", marker="o", linestyle="-", label="Lookahead")
axes[1,0].set_title("QFT: Fidelities")
axes[1,0].set_ylabel("Fidelity")
axes[1,0].set_xlabel("Circuit index")
axes[1,0].set_xticks([])  # remove x ticks
axes[1,0].legend()

axes[1,1].plot(idx, qft_improve_base, "o-", label="Baseline")
axes[1,1].plot(idx, qft_improve_new, "o-", color="green", marker="o", linestyle="-", label="Lookahead")
axes[1,1].set_title("QFT: Per-Circuit Improvement")
axes[1,1].set_ylabel("Δ Fidelity")
axes[1,1].set_xlabel("Circuit index")
axes[1,1].set_xticks([])  # remove x ticks
axes[1,1].legend()

axes[1,2].bar(["Baseline", "Lookahead"],
              [qft_improve_base.mean(), qft_improve_new.mean()],
              color=["C0", "green"])
axes[1,2].set_title("QFT: Avg. Improvement")
axes[1,2].set_ylabel("Mean Δ Fidelity")

fig.suptitle("Comparison of Baseline vs Lookahead\nRandom Circuits and QFT", fontsize=16)
plt.tight_layout()
plt.show()
