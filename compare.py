import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 0.1
})

def annotate_points(ax, x, y, offset=0.0):
    for i, val in enumerate(y):
        ax.text(x[i], y[i] + offset, f'{val:.1f}', ha='center', va='bottom', fontsize=11)

def main():
    drl_df = pd.read_excel("DRL.xlsx")
    edf_df = pd.read_excel("edf.xlsx")

    merged = pd.merge(drl_df, edf_df, on="Class", suffixes=("_DRL", "_EDF"))
    classes = merged["Class"]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 5))
    y_drl = merged["SuccessRate_DRL"]
    y_edf = merged["SuccessRate_EDF"]

    y_drl_min = merged["SuccessRate_min_DRL"]
    y_drl_max = merged["SuccessRate_max_DRL"]
    drl_err = [y_drl - y_drl_min, y_drl_max - y_drl]

    y_edf_min = merged["SuccessRate_min_EDF"]
    y_edf_max = merged["SuccessRate_max_EDF"]
    edf_err = [y_edf - y_edf_min, y_edf_max - y_edf]

    ax.plot(x, y_drl, marker='o', label="DRL", color='tab:blue')
    ax.plot(x, y_edf, marker='o', linestyle='--', label="EDF", color='tab:orange')

    ax.errorbar(x, y_drl, yerr=drl_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:blue')
    ax.errorbar(x, y_edf, yerr=edf_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:orange')

    y_min = min(y_drl.min(), y_edf.min())
    y_max = max(y_drl.max(), y_edf.max())
    y_range = y_max - y_min
    offset = 0.02 * y_range
    annotate_points(ax, x, y_drl, 1.2*offset)
    annotate_points(ax, x, y_edf, 1.2*offset)

    ax.set_ylim(top=y_max + 5*offset)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Scenario Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Figure_sucuss_rate.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    y_drl = merged["AvgLatency_DRL"]
    y_edf = merged["AvgLatency_EDF"]

    y_drl_min = merged["AvgLatency_min_DRL"]
    y_drl_max = merged["AvgLatency_max_DRL"]
    drl_err = [y_drl - y_drl_min, y_drl_max - y_drl]

    y_edf_min = merged["AvgLatency_min_EDF"]
    y_edf_max = merged["AvgLatency_max_EDF"]
    edf_err = [y_edf - y_edf_min, y_edf_max - y_edf]

    ax.plot(x, y_drl, marker='o', label="DRL", color='tab:blue')
    ax.plot(x, y_edf, marker='o', linestyle='--', label="EDF", color='tab:orange')

    ax.errorbar(x, y_drl, yerr=drl_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:blue')
    ax.errorbar(x, y_edf, yerr=edf_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:orange')

    y_min = min(y_drl.min(), y_edf.min())
    y_max = max(y_drl.max(), y_edf.max())
    y_range = y_max - y_min
    offset = 0.02 * y_range
    annotate_points(ax, x, y_drl, offset)
    annotate_points(ax, x, y_edf, offset)

    ax.set_ylim(top=y_max + 24*offset)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Delay (ms)")
    ax.set_xlabel("Scenario Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Figure_mean_delay.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    y_drl = merged["IdleTime_DRL"]
    y_edf = merged["IdleTime_EDF"]

    y_drl_min = merged["IdleTime_min_DRL"]
    y_drl_max = merged["IdleTime_max_DRL"]
    drl_err = [y_drl - y_drl_min, y_drl_max - y_drl]

    y_edf_min = merged["IdleTime_min_EDF"]
    y_edf_max = merged["IdleTime_max_EDF"]
    edf_err = [y_edf - y_edf_min, y_edf_max - y_edf]

    ax.plot(x, y_drl, marker='o', label="DRL", color='tab:blue')
    ax.plot(x, y_edf, marker='o', linestyle='--', label="EDF", color='tab:orange')

    ax.errorbar(x, y_drl, yerr=drl_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:blue')
    ax.errorbar(x, y_edf, yerr=edf_err, fmt='none', capsize=5, alpha=0.5, ecolor='tab:orange')

    y_min = min(y_drl.min(), y_edf.min())
    y_max = max(y_drl.max(), y_edf.max())
    y_range = y_max - y_min
    offset = 0.02 * y_range

    annotate_points(ax, x, y_drl, -5*offset)
    annotate_points(ax, x, y_edf, 2*offset)

    ax.set_ylim(top=y_max + 6*offset)
    ax.set_ylim(bottom=y_min - 5*offset)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Idle (%)")
    ax.set_xlabel("Scenario Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Figure_idle_per.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Comparison charts saved.")

if __name__ == "__main__":
    main()
