"""Descriptive mean predicted dose-response plots, not causal validation."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_dose_response(data, output_dir):
    for split, rows in data.groupby("split"):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for name, group in rows.groupby("model"):
            means = group.groupby("dose").response.mean()
            ax.plot(means.index, means.values, label=name)
        ax.set(
            xlabel="Dose",
            ylabel="Mean predicted response",
            title=f"{split}: model-based dose-response",
        )
        ax.legend(fontsize=8)
        fig.savefig(Path(output_dir) / f"{split}_dose_response.png", dpi=150)
        plt.close(fig)
