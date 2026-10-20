"""Generate a deterministic scorecard for lab 05½.

Writes ../figures/evaluation_scorecard.png and ../results/evaluation_lab.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import (
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    metrics_by_slice,
    split_indices,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
N_RESAMPLES = 2000


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 160
    group = np.where(np.arange(n) % 4 == 0, "low-light", "daylight")
    latent = rng.normal(0, 1, n) + np.where(group == "daylight", 0.35, -0.1)
    truth = (latent + rng.normal(0, 0.65, n) > 0).astype(int)
    probability = 1 / (1 + np.exp(-(latent * 1.8 + rng.normal(0, 0.8, n))))
    return truth, probability, group


def gap_ci(truth, probability, group, a: str, b: str, seed: int = 1):
    """95% interval for F1(a) - F1(b), resampling each slice on its own."""
    rng = np.random.default_rng(seed)
    ia, ib = np.flatnonzero(group == a), np.flatnonzero(group == b)
    gaps = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        ra, rb = rng.choice(ia, len(ia)), rng.choice(ib, len(ib))
        gaps[i] = (classification_metrics(truth[ra], probability[ra])["f1"]
                   - classification_metrics(truth[rb], probability[rb])["f1"])
    low, high = np.quantile(gaps, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    truth, probability, group = make_data()
    results: dict = {"n_examples": int(len(truth)), "n_resamples": N_RESAMPLES}

    # 1. How the threshold moves the trade-off, on all 160 rows (illustration only).
    print("All 160 rows, three thresholds (illustration, not a reported result)")
    print("threshold  accuracy  precision  recall    f1")
    results["sweep_all_rows"] = {}
    for threshold in (0.3, 0.5, 0.7):
        m = classification_metrics(truth, probability, threshold)
        results["sweep_all_rows"][f"{threshold:.2f}"] = {
            k: round(m[k], 4) for k in ("accuracy", "precision", "recall", "f1")
        }
        print(
            f"{threshold:>8.2f}  {m['accuracy']:>8.3f}  {m['precision']:>9.3f}"
            f"  {m['recall']:>6.3f}  {m['f1']:>5.3f}"
        )

    # 2. The honest protocol: choose on validation, report once on test.
    train, validation, test = split_indices(len(truth), seed=0)
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    val_f1 = [classification_metrics(truth[validation], probability[validation], t)["f1"]
              for t in grid]
    chosen = float(grid[int(np.argmax(val_f1))])  # argmax keeps the smallest on ties
    test_f1, test_low, test_high = bootstrap_ci(
        truth[test], probability[test], "f1", chosen, n_resamples=N_RESAMPLES
    )
    results["protocol"] = {
        "split_train_validation_test": [len(train), len(validation), len(test)],
        "threshold_chosen_on_validation": chosen,
        "validation_f1": round(max(val_f1), 4),
        "test_f1": round(test_f1, 4),
        "test_f1_ci95": [round(test_low, 4), round(test_high, 4)],
    }
    print(f"\nSplit {len(train)}/{len(validation)}/{len(test)}: threshold {chosen:.2f} "
          f"chosen on validation (F1 {max(val_f1):.3f}); "
          f"test F1 {test_f1:.3f}, 95% CI [{test_low:.3f}, {test_high:.3f}]")

    # 3. Calibration and slices, on all 160 rows at 0.5, with bootstrap intervals.
    ece = expected_calibration_error(truth, probability)
    results["ece_10_bins_all_rows"] = round(ece, 4)
    print(f"\nECE (10 bins): {ece:.3f}")
    print("Slices at threshold 0.5, F1 with 95% bootstrap interval:")
    slice_scores = metrics_by_slice(truth, probability, group)
    intervals = {}
    results["slices_all_rows"] = {}
    for label, metrics in slice_scores.items():
        mask = group == label
        f1, low, high = bootstrap_ci(truth[mask], probability[mask], "f1",
                                     n_resamples=N_RESAMPLES)
        intervals[label] = (low, high)
        results["slices_all_rows"][label] = {
            "n": int(mask.sum()), "positives": int(truth[mask].sum()),
            "recall": round(metrics["recall"], 4), "f1": round(f1, 4),
            "f1_ci95": [round(low, 4), round(high, 4)],
        }
        print(f"  {label:>9}: n={mask.sum():>3}, recall={metrics['recall']:.3f}, "
              f"f1={f1:.3f} [{low:.3f}, {high:.3f}]")
    gap = slice_scores["daylight"]["f1"] - slice_scores["low-light"]["f1"]
    gap_low, gap_high = gap_ci(truth, probability, group, "daylight", "low-light")
    results["f1_gap_daylight_minus_lowlight"] = {
        "gap": round(gap, 4), "ci95": [round(gap_low, 4), round(gap_high, 4)],
    }
    print(f"  F1 gap daylight - low-light: {gap:+.3f}, "
          f"95% CI [{gap_low:+.3f}, {gap_high:+.3f}]")

    curves = {
        key: [classification_metrics(truth, probability, t)[key] for t in grid]
        for key in ("precision", "recall", "f1")
    }

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for label, values in curves.items():
        axes[0].plot(grid, values, label=label)
    axes[0].axvline(0.5, color="#2b2b2b", linestyle="--", linewidth=1)
    axes[0].set(xlabel="decision threshold", ylabel="score", ylim=(0, 1))
    axes[0].set_title("The threshold changes the trade-off")
    axes[0].legend(frameon=False)

    labels = list(slice_scores)
    x = np.arange(len(labels))
    f1s = np.array([slice_scores[label]["f1"] for label in labels])
    err = np.array([[f1s[i] - intervals[label][0], intervals[label][1] - f1s[i]]
                    for i, label in enumerate(labels)]).T
    axes[1].bar(x, f1s, width=0.5, color="#1f918d", yerr=err, capsize=8,
                error_kw={"elinewidth": 1.5})
    for i, label in enumerate(labels):
        n = results["slices_all_rows"][label]["n"]
        axes[1].text(x[i], 0.05, f"F1 {f1s[i]:.2f}\nn = {n}", ha="center",
                     color="white", fontsize=9)
    axes[1].set_xticks(x, labels)
    axes[1].set(ylim=(0, 1), ylabel="F1 at 0.5, 95% bootstrap interval")
    axes[1].set_title("Slices need error bars: 40 low-light examples\n"
                      "cannot tell 0.73 from 0.81", fontsize=11)
    fig.tight_layout()
    path = FIGURES / "evaluation_scorecard.png"
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "evaluation_lab.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
