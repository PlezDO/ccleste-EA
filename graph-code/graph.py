import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("../graphs", exist_ok=True)

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


def plot_param_sweep(df, fixed, vary_col, vary_vals,
                     fitness_col, ylabel, title, filename):
    """
    Plot mean ± std of `fitness_col` across generations for each value in
    `vary_vals`, holding all other params fixed at `fixed`.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in fixed.items():
        if col != vary_col:
            mask &= (df[col] == val)
    filtered = df[mask]

    plt.figure(figsize=(10, 6))
    for val in vary_vals:
        subset  = filtered[filtered[vary_col] == val]
        grouped = subset.groupby("generation")[fitness_col]
        mean    = grouped.mean()
        std     = grouped.std()
        gens    = mean.index

        label = f"{vary_col}={val}"
        if val == default_params[vary_col]: 
            plt.plot(gens, mean, label=label + " (default)", linewidth=2.5)
        else:
            plt.plot(gens, mean, label=label)
        plt.fill_between(gens, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../graphs/" + filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


plot_param_sweep(df, default_params, "N", SWEEP_N,
                 "best_fitness", "Best Fitness",
                 "Effect of Population Size on Best Fitness",
                 "plot_N_best.png")

plot_param_sweep(df, default_params, "N", SWEEP_N,
                 "avg_fitness", "Average Fitness",
                 "Effect of Population Size on Average Fitness",
                 "plot_N_avg.png")

plot_param_sweep(df, default_params, "p_m", SWEEP_PM,
                 "best_fitness", "Best Fitness",
                 "Effect of Mutation Probability on Best Fitness",
                 "plot_pm_best.png")

plot_param_sweep(df, default_params, "p_m", SWEEP_PM,
                 "avg_fitness", "Average Fitness",
                 "Effect of Mutation Probability on Average Fitness",
                 "plot_pm_avg.png")

plot_param_sweep(df, default_params, "p_c", SWEEP_PC,
                 "best_fitness", "Best Fitness",
                 "Effect of Uniform Crossover Probability on Best Fitness",
                 "plot_pc_best.png")

plot_param_sweep(df, default_params, "p_c", SWEEP_PC,
                 "avg_fitness", "Average Fitness",
                 "Effect of Uniform Crossover Probability on Average Fitness",
                 "plot_pc_avg.png")

plot_param_sweep(df, default_params, "trn_size", SWEEP_TRN_SIZE,
                 "best_fitness", "Best Fitness",
                 "Effect of Tournament Size on Best Fitness",
                 "plot_trn_best.png")

plot_param_sweep(df, default_params, "trn_size", SWEEP_TRN_SIZE,
                 "avg_fitness", "Average Fitness",
                 "Effect of Tournament Size on Average Fitness",
                 "plot_trn_avg.png")

last_gen = df["generation"].max()

hm1 = df[
    (df["N"]        == default_params["N"]) &
    (df["trn_size"] == default_params["trn_size"]) &
    (df["generation"] == last_gen)
].groupby(["p_m", "p_c"])["best_fitness"].mean().unstack()

plt.figure(figsize=(8, 6))
sns.heatmap(hm1, annot=True, fmt=".2f", cmap="viridis")
plt.ylabel("Mutation Probability (p_m)")
plt.xlabel("Crossover Probability (p_c)")
plt.title(f"Mean Best Fitness at Generation {last_gen}\n(N={default_params['N']}, trn={default_params['trn_size']})")
plt.tight_layout()
plt.savefig("../graphs/heatmap_pm_pc.png", dpi=150)
plt.close()
print("Saved heatmap_pm_pc.png")

hm2 = df[
    (df["p_m"] == default_params["p_m"]) &
    (df["p_c"] == default_params["p_c"]) &
    (df["generation"] == last_gen)
].groupby(["N", "trn_size"])["best_fitness"].mean().unstack()

plt.figure(figsize=(8, 6))
sns.heatmap(hm2, annot=True, fmt=".2f", cmap="viridis")
plt.ylabel("Population Size (N)")
plt.xlabel("Tournament Size")
plt.title(f"Mean Best Fitness at Generation {last_gen}\n(p_m={default_params['p_m']}, p_c={default_params['p_c']})")
plt.tight_layout()
plt.savefig("../graphs/heatmap_N_trn.png", dpi=150)
plt.close()
print("Saved heatmap_N_trn.png")

print("\nAll plots saved.")
