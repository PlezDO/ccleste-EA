import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("../graphs", exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

LINE_STYLES = ['-', '-', '-', '-']
SMOOTH = 10

df = pd.read_csv("../graph-data/master_results.csv")

default_params = {
    "N":        300,
    "p_m":      0.05,
    "p_c":      0.02,
    "trn_size": 20,
}

SWEEP_N        = [100, 200, 300, 500]
SWEEP_PM       = [0.01, 0.03, 0.05, 0.08]
SWEEP_PC       = [0.0,  0.01, 0.02, 0.05]
SWEEP_TRN_SIZE = [5,    10,   20,   35]


def plot_combined(df, fixed, vary_col, vary_vals, title, filename):
    mask = pd.Series(True, index=df.index)
    for col, val in fixed.items():
        if col != vary_col:
            mask &= (df[col] == val)
    filtered = df[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    for i, val in enumerate(vary_vals):
        subset  = filtered[filtered[vary_col] == val]
        grouped = subset.groupby("generation")
        best    = grouped["best_fitness"].mean().rolling(SMOOTH, min_periods=1).mean()
        avg     = grouped["avg_fitness"].mean().rolling(SMOOTH, min_periods=1).mean()
        gens    = best.index
        label   = f"{vary_col}={val}" + (" (default)" if val == default_params[vary_col] else "")
        lw      = 2.5 if val == default_params[vary_col] else 1.5

        ax1.plot(gens, best, label=label, linewidth=lw)
        ax2.plot(gens, avg,  label=label, linewidth=lw)

    for ax, ylabel in zip([ax1, ax2], ["Best Fitness", "Average Fitness"]):
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    ax1.set_title("Best Fitness")
    ax2.set_title("Average Fitness")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig("../graphs/" + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


plot_combined(df, default_params, "N", SWEEP_N,
              "Effect of Population Size", "plot_N.png")

plot_combined(df, default_params, "p_m", SWEEP_PM,
              "Effect of Mutation Probability", "plot_pm.png")

plot_combined(df, default_params, "p_c", SWEEP_PC,
              "Effect of Crossover Probability", "plot_pc.png")

plot_combined(df, default_params, "trn_size", SWEEP_TRN_SIZE,
              "Effect of Tournament Size", "plot_trn.png")


last_gen = df["generation"].max()

hm1 = df[
    (df["N"]        == default_params["N"]) &
    (df["trn_size"] == default_params["trn_size"]) &
    (df["generation"] == last_gen)
].groupby(["p_m", "p_c"])["best_fitness"].mean().unstack()

plt.figure(figsize=(3.5, 2.5))
sns.heatmap(hm1, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 7})
plt.ylabel("Mutation Probability (p_m)")
plt.xlabel("Crossover Probability (p_c)")
plt.title(f"Mean Best Fitness at Generation {last_gen}\n(N={default_params['N']}, trn={default_params['trn_size']})")
plt.tight_layout()
plt.savefig("../graphs/heatmap_pm_pc.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved heatmap_pm_pc.png")

hm2 = df[
    (df["p_m"] == default_params["p_m"]) &
    (df["p_c"] == default_params["p_c"]) &
    (df["generation"] == last_gen)
].groupby(["N", "trn_size"])["best_fitness"].mean().unstack()

plt.figure(figsize=(3.5, 2.5))
sns.heatmap(hm2, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 7})
plt.ylabel("Population Size (N)")
plt.xlabel("Tournament Size")
plt.title(f"Mean Best Fitness at Generation {last_gen}\n(p_m={default_params['p_m']}, p_c={default_params['p_c']})")
plt.tight_layout()
plt.savefig("../graphs/heatmap_N_trn.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved heatmap_N_trn.png")


params_and_sweeps = [
    ("N",        SWEEP_N,        "Population Size (N)"),
    ("p_m",      SWEEP_PM,       "Mutation Probability (p_m)"),
    ("p_c",      SWEEP_PC,       "Crossover Probability (p_c)"),
    ("trn_size", SWEEP_TRN_SIZE, "Tournament Size"),
]

fig, axes = plt.subplots(2, 2, figsize=(7, 5))

for ax, (param_col, param_vals, xlabel) in zip(axes.flat, params_and_sweeps):
    positive_counts = []
    for val in param_vals:
        mask = pd.Series(True, index=df.index)
        for col, default_val in default_params.items():
            if col != param_col:
                mask &= (df[col] == default_val)
        subset = df[mask & (df[param_col] == val)]
        positive_counts.append((subset["best_fitness"] > 0).sum())

    bars = ax.bar([str(v) for v in param_vals], positive_counts, color='steelblue')
    ax.bar_label(bars)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(positive_counts) * 1.15 if max(positive_counts) > 0 else 1)
    ax.set_title(f"Positive Fitness by {xlabel}")

fig.suptitle("Generations with Positive Best Fitness", fontsize=8)
plt.tight_layout()
plt.savefig("../graphs/outliers.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved outliers.png")

print("\nAll plots saved.")
