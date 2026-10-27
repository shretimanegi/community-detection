"""
Regenerates output/results.md from output/results.csv (the same source
plot_results.py reads). Called automatically at the end of run.py; run it
standalone any time to refresh the table without re-running detection.

Layout: two rows per community (one for Modularity, one for NMI), one
column per algorithm -- so the two scores land on separate table rows
instead of being stacked inside a single cell. The best-scoring
algorithm's value for each metric is bolded within its row (modularity
and NMI are judged independently, so a different algorithm can win each
one).
"""

import os

from results_store import load_results, OUTPUT_DIR
from datasets import DATASETS

# Datasets whose ground truth we invented ourselves (see datasets.py
# docstrings) get a "*" marker in the table, explained in the footnote.
PROXY_DATASETS = {
    name for name, loader in DATASETS.items()
    if loader.__doc__ and "PROXY ground truth" in loader.__doc__
}


def fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


def generate():
    results = load_results()
    communities = list(dict.fromkeys(r["community"] for r in results))
    algorithms = list(dict.fromkeys(r["algorithm"] for r in results))

    def score(community, algorithm, metric):
        for r in results:
            if r["community"] == community and r["algorithm"] == algorithm:
                return r[metric]
        return None

    lines = []
    lines.append("# Community Detection Results")
    lines.append("")
    lines.append("Each community/dataset spans two rows -- Modularity then NMI -- one column per "
                  "algorithm. **bold** marks the best-scoring algorithm for that metric's row "
                  "(modularity and NMI are judged separately, so a different algorithm can win each).")
    lines.append("")

    lines.append("| Community | Metric | " + " | ".join(algorithms) + " |")
    lines.append("|---|---|" + "---|" * len(algorithms))

    for community in communities:
        label = community + ("*" if community in PROXY_DATASETS else "")

        mod_values = {a: score(community, a, "modularity") for a in algorithms}
        nmi_values = {a: score(community, a, "nmi") for a in algorithms}

        best_mod = max((v for v in mod_values.values() if v is not None), default=None)
        best_nmi = max((v for v in nmi_values.values() if v is not None), default=None)

        mod_cells = []
        for a in algorithms:
            mod_str = fmt(mod_values[a])
            if mod_values[a] is not None and best_mod is not None and mod_values[a] == best_mod:
                mod_str = f"**{mod_str}**"
            mod_cells.append(mod_str)

        nmi_cells = []
        for a in algorithms:
            nmi_str = fmt(nmi_values[a])
            if nmi_values[a] is not None and best_nmi is not None and nmi_values[a] == best_nmi:
                nmi_str = f"**{nmi_str}**"
            nmi_cells.append(nmi_str)

        lines.append(f"| {label} | Modularity | " + " | ".join(mod_cells) + " |")
        lines.append(f"|  | NMI | " + " | ".join(nmi_cells) + " |")

    lines.append("")
    lines.append("\\* No built-in ground truth exists for these datasets. NMI is computed against a proxy ground truth defined from external knowledge, not the graph itself:")
    lines.append("- **Florentine Families**: historical Medici party vs. Albizzi-led oligarch party (Padgett & Ansell, 1993) — a documented split, though a few families' allegiance is debated.")
    lines.append("- **Dolphins**: the first two-way split of Girvan-Newman edge-betweenness clustering (21 vs 41 dolphins), a common stand-in for Lusseau's field-observed groups, which are not in the data file. It is computed from the graph's own edges, so NMI against it is circular and favours structure-based methods.")
    lines.append("- **Yeast**: the Girvan-Newman edge-betweenness dendrogram cut at its maximum-modularity level (39 communities). The data file has anonymized node IDs and no functional labels. The labels are computed from the graph's own edges, so NMI against them is circular and favours modularity-driven methods.")
    lines.append("")
    lines.append("Treat these NMI values as illustrative, with lower confidence than Karate Club, American College Football, and Cora (which all have a real, established ground truth).")
    lines.append("")
    lines.append("Summary bar charts across all runs: `output/modularity_histogram.png`, `output/nmi_histogram.png`.")
    lines.append("")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    generate()
