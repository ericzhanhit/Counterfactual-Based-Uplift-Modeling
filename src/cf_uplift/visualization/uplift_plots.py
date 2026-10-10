"""Plot saved observed-data rankings without training models."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..metrics.uplift import cumulative_curves


def plot_predictions(predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, data in predictions.groupby("split"):
        for metric in ["uplift", "qini"]:
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            for name, rows in data.groupby("model"):
                rows = rows.sort_values("row_id", kind="stable")
                curve = cumulative_curves(rows.outcome, rows.treatment, rows.uplift)
                ax.plot(
                    curve.population_fraction * 100,
                    curve[metric],
                    label=name,
                    linewidth=1.4,
                )
            # The endpoint is invariant to ranking and provides the random baseline.
            ax.plot(
                [0, 100],
                [0, curve[metric].iloc[-1]],
                "k--",
                label="Endpoint random line",
            )
            ax.set(
                xlabel="Population targeted (%)",
                ylabel=f"Raw {metric} gain",
                title=f"{split}: observed outcomes",
            )
            ax.legend(fontsize=7, ncol=2)
            fig.savefig(output_dir / f"{split}_{metric}.png", dpi=150)
            plt.close(fig)
