"""Generate a deterministic scorecard for lab 05½."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import (
    classification_metrics,
    expected_calibration_error,
    metrics_by_slice,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 160
    group = np.where(np.arange(n) % 4 == 0, "low-light", "daylight")
    latent = rng.normal(0, 1, n) + np.where(group == "daylight", 0.35, -0.1)
    truth = (latent + rng.normal(0, 0.65, n) > 0).astype(int)
    probability = 1 / (1 + np.exp(-(latent * 1.8 + rng.normal(0, 0.8, n))))
    return truth, probability, group


def main() -> None:
    truth, probability, group = make_data()
    thresholds = [0.3, 0.5, 0.7]
    print("threshold  accuracy  precision  recall    f1")
    for threshold in thresholds:
        m = classification_metrics(truth, probability, threshold)
        print(
            f"{threshold:>8.2f}  {m['accuracy']:>8.3f}  {m['precision']:>9.3f}"
            f"  {m['recall']:>6.3f}  {m['f1']:>5.3f}"
        )

    print(f"\nECE (10 bins): {expected_calibration_error(truth, probability):.3f}")
    print("Slices at threshold 0.5:")
    for label, metrics in metrics_by_slice(truth, probability, group).items():
        print(f"  {label:>9}: recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")

    grid = np.linspace(0.05, 0.95, 91)
    curves = {
        key: [classification_metrics(truth, probability, t)[key] for t in grid]
        for key in ("precision", "recall", "f1")
    }
    slice_scores = metrics_by_slice(truth, probability, group)

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, values in curves.items():
        axes[0].plot(grid, values, label=label)
    axes[0].axvline(0.5, color="#2b2b2b", linestyle="--", linewidth=1)
    axes[0].set(xlabel="decision threshold", ylabel="score", ylim=(0, 1))
    axes[0].set_title("The threshold changes the trade-off")
    axes[0].legend(frameon=False)

    labels = list(slice_scores)
    x = np.arange(len(labels))
    axes[1].bar(
        x - 0.18,
        [slice_scores[label]["recall"] for label in labels],
        width=0.36,
        label="recall",
    )
    axes[1].bar(
        x + 0.18,
        [slice_scores[label]["f1"] for label in labels],
        width=0.36,
        label="F1",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("The aggregate hides slice behavior")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = FIGURES / "evaluation_scorecard.png"
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
