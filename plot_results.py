"""
Summary histograms across every (community, algorithm) run recorded in
output/results.csv: output/modularity_histogram.png and
output/nmi_histogram.png. Called automatically at the end of run.py; run it
standalone any time to refresh the plots without re-running detection.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from results_store import load_results, OUTPUT_DIR


# Algorithms left out of the charts only -- they stay in results.csv and the
# results.md table. Empty this set to bring them back.
HIDDEN_FROM_PLOTS = {"Louvain", "Leiden", "Infomap", "LPA-MNI+ML"}


def _score(results, community, algorithm, metric):
    for r in results:
        if r["community"] == community and r["algorithm"] == algorithm:
            return r[metric]
    return None


def _plot_metric(results, communities, algorithms, metric, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    n_algos = len(algorithms)
    bar_width = 0.8 / n_algos
    x = np.arange(len(communities))

    for i, algo in enumerate(algorithms):
        values = [_score(results, c, algo, metric) for c in communities]
        heights = [v if v is not None else 0 for v in values]
        offset = (i - (n_algos - 1) / 2) * bar_width
        bars = ax.bar(x + offset, heights, width=bar_width, label=algo)

        for rect, v in zip(bars, values):
            if v is not None:
                ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(communities, rotation=20, ha="right")
    ax.set_ylabel(metric.upper() if metric == "nmi" else metric.capitalize())
    ax.set_title(title)
    ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate():
    results = load_results()
    communities = list(dict.fromkeys(r["community"] for r in results))
    algorithms = [a for a in dict.fromkeys(r["algorithm"] for r in results)
                  if a not in HIDDEN_FROM_PLOTS]

    _plot_metric(results, communities, algorithms, "modularity",
                 "Modularity by Community and Algorithm", "modularity_histogram.png")
    _plot_metric(results, communities, algorithms, "nmi",
                 "NMI by Community and Algorithm", "nmi_histogram.png")


if __name__ == "__main__":
    generate()
