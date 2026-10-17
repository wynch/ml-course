"""Every figure of module 13, drawn from results/*.json alone (no model loaded).

    uv run python src/figures.py
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import FIGURES, read_results  # noqa: E402

SET_LABEL = {"V": "drawn V", "O": "drawn O", "cat": "cats", "car": "CIFAR cars",
             "truck": "garbage trucks", "imagenette": "other imagenette"}
COLOURS = {"V": "#b8336a", "O": "#7a7a7a", "cat": "#d08c1a", "car": "#1f6fb2",
           "truck": "#2c8c5a", "imagenette": "#6d5bb5"}


def _img(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _save(fig, name: str, dpi: int = 120) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_v_neuron() -> None:
    r = read_results("v_neuron")
    acts = read_results("v_neuron_acts")
    j = r["channels"].index(r["target"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    rng = np.random.default_rng(0)
    for a, key, title in ((ax[0], "trained", "trained weights"), (ax[1], "random_55", "random weights")):
        for i, s in enumerate(r["sets"]):
            vals = np.array(acts[key][s][j] if key == "trained" else acts[key][s])
            a.scatter(i + rng.uniform(-0.28, 0.28, len(vals)), vals, s=5, alpha=0.35, color=COLOURS[s])
            med = np.median(vals)
            a.plot([i - 0.36, i + 0.36], [med, med], color="black", lw=2)
        a.set_xticks(range(len(r["sets"])), [SET_LABEL[s] for s in r["sets"]], rotation=25, ha="right")
        a.axhline(0, color="black", lw=0.5)
        a.set_ylabel("4e:55, max over the map (before ReLU)")
        a.set_title(f"4e:55, {title} (bar = median)")
    g = r["trained"]["gap_by_channel"]
    chans = r["channels"]
    ax[2].bar(chans, [g[str(c)] for c in chans],
              color=["#b8336a" if c == r["target"] else "#9aa5a2" for c in chans])
    ax[2].axhline(0, color="black", lw=0.5)
    ax[2].set_xlabel("channel of mixed4e")
    ax[2].set_ylabel("gap, in pooled SD")
    ax[2].set_title(f"V minus max(cats, cars): 4e:55 ranks {r['trained']['rank_55']} of 21")
    fig.tight_layout()
    _save(fig, "v_neuron.png")


def fig_v_examples() -> None:
    r = read_results("v_neuron_acts")
    sets = r["sets"]
    k = len(r["thumbs"][sets[0]]) // 2
    fig, ax = plt.subplots(len(sets), 2 * k, figsize=(2 * k * 1.05, len(sets) * 1.25))
    for row, s in enumerate(sets):
        for c, t in enumerate(r["thumbs"][s]):
            a = ax[row, c]
            a.imshow(_img(t["jpg"]))
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{t['a']:.0f}", fontsize=7, pad=2)
        ax[row, 0].set_ylabel(SET_LABEL[s], fontsize=8)
    fig.suptitle(f"4e:55 on each set: the {k} highest (left) and {k} lowest (right) images, "
                 "activation above each", fontsize=10)
    fig.tight_layout()
    _save(fig, "v_examples.png", dpi=80)


def fig_feature_vis() -> None:
    r = read_results("feature_vis")
    n = len(r["renders"])
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    for a, rd in zip(ax, r["renders"]):
        a.imshow(_img(rd["jpg"]))
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"4e:{rd['channel']}: {rd['activation']:.0f}\n(best real image {rd['best_real_image']:.0f})",
                    fontsize=9)
    fig.suptitle(f"Lucent renders, {r['steps']} steps from noise, {r['size']} px", fontsize=10)
    fig.tight_layout()
    _save(fig, "feature_vis.png", dpi=80)


def fig_clip_probe() -> None:
    r = read_results("clip")["probe"]
    L = np.arange(len(r["accuracy_trained"]))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(L, r["accuracy_trained"], "o-", color="#b8336a", label="trained CLIP")
    ax.plot(L, r["accuracy_random_weights"], "s--", color="#7a7a7a", label="random weights")
    ax.axhline(r["chance"], color="black", lw=0.6, ls=":", label="chance (10 classes)")
    ax.set_xticks(L, ["emb"] + [str(i) for i in L[1:]])
    ax.set_xlabel("after block")
    ax.set_ylabel("held-out accuracy (500 images)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A linear probe on the CLS vector, imagenette")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "clip_probe.png")


def fig_clip_neurons() -> None:
    r = read_results("clip")
    ns = r["neurons"]
    top = len(ns[0]["top"])
    fig, ax = plt.subplots(len(ns), top, figsize=(top * 1.35, len(ns) * 1.7))
    for row, n in enumerate(ns):
        for c, t in enumerate(n["top"]):
            a = ax[row, c]
            a.imshow(_img(t["jpg"]))
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{t['class']}\n{t['activation']:.2f}", fontsize=7, pad=2)
        ax[row, 0].set_ylabel(n["id"], fontsize=9)
    fig.suptitle("Three CLIP MLP neurons drawn at random: the 8 images (of 1,500) that drive each hardest",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "clip_neurons.png", dpi=80)


def fig_jepa_probe() -> None:
    r = read_results("jepa_probe")
    p = r["probes"]
    names = ["trained", "random_0", "random_1", "random_2", "raw_window"]
    labels = ["trained", "random 0", "random 1", "random 2", "raw window"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    w = 0.26
    x = np.arange(len(names))
    for i, (m, lab, col) in enumerate((("position_r2", "position R²", "#1f6fb2"),
                                       ("adjacency_acc", "8 neighbours, accuracy", "#2c8c5a"),
                                       ("path_r2", "path length R²", "#b8336a"))):
        vals = [p[n][m] for n in names]
        ax[0].bar(x + (i - 1) * w, vals, w, label=lab, color=col)
    ax[0].set_xticks(x, labels)
    lo = min(min(p[n][m] for n in names) for m in ("position_r2", "adjacency_acc", "path_r2"))
    ax[0].set_ylim(min(0, lo) - 0.1, 1.05)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax[0].set_title(f"Linear probes on the frozen latent, {r['n_test']:,} held-out frames")
    s = r["scatter"]
    ax[1].scatter(s["true"], s["trained"], s=7, alpha=0.5, color="#b8336a", label="trained")
    ax[1].scatter(s["true"], s["random_0"], s=7, alpha=0.4, color="#7a7a7a", label="random 0")
    ax[1].plot([0, r["cap"]], [0, r["cap"]], "k--", lw=0.8)
    ax[1].set_xlabel("true steps to goal (capped at 30)")
    ax[1].set_ylabel("probe's guess")
    ax[1].set_title(f"Path length: R² {p['trained']['path_r2']:.2f} trained, "
                    f"{p['random_0']['path_r2']:.2f} random 0")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "jepa_probe.png")


def main() -> None:
    fig_v_neuron()
    fig_v_examples()
    fig_feature_vis()
    fig_clip_probe()
    fig_clip_neurons()
    fig_jepa_probe()


if __name__ == "__main__":
    main()
