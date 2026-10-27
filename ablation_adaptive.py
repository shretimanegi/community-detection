"""Ablation: ML seed vs LPA-MNI+ML vs adaptive Phase 2 vs adaptive-from-singletons vs Louvain."""
import time

import networkx as nx
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score

from algorithms.lpa_mni import multilevel_seed_communities, labels_to_communities
from algorithms.lpa_mni_adaptive import phase2
from datasets import DATASETS


def score(G, labels, gt):
    q = modularity(G, labels_to_communities(labels))
    nodes = sorted(G.nodes())
    nmi = (normalized_mutual_info_score([gt[n] for n in nodes], [labels[n] for n in nodes])
           if gt else float("nan"))
    return len(set(labels.values())), q, nmi


rows = []
for name, load in DATASETS.items():
    G, gt, _ = load()
    t0 = time.perf_counter(); seed = multilevel_seed_communities(G); ts = time.perf_counter() - t0
    rows.append((name, "A. ML seed only", *score(G, seed, gt), ts, ""))
    for m, ad in [("B. LPA-MNI+ML (majority/DC)", False), ("C. ML seed + adaptive P2", True)]:
        lab, t, upd, roles = phase2(G, seed, adaptive=ad)
        rows.append((name, m, *score(G, lab, gt), ts + t, dict(roles) if ad else ""))
    lab, t, upd, _ = phase2(G, {u: u for u in G}, adaptive=True)
    rows.append((name, "D. Adaptive LPA from singletons", *score(G, lab, gt), t, ""))
    t0 = time.perf_counter(); lv = nx.community.louvain_communities(G, seed=42); t = time.perf_counter() - t0
    lvl = {u: i for i, g in enumerate(lv) for u in g}
    rows.append((name, "E. Louvain (networkx)", *score(G, lvl, gt), t, ""))

print(f"{'Dataset':26s}{'Method':34s}{'k':>5s}{'Q':>8s}{'NMI':>8s}{'time(s)':>9s}  roles")
prev = None
for r in rows:
    if prev and r[0] != prev: print()
    prev = r[0]
    print(f"{r[0]:26s}{r[1]:34s}{r[2]:5d}{r[3]:8.4f}{r[4]:8.4f}{r[5]:9.3f}  {r[6]}")

# ---- charts ----
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
datasets = list(dict.fromkeys(r[0] for r in rows))
methods = list(dict.fromkeys(r[1] for r in rows))
for idx, label, fname in [(3, "Modularity", "ablation_modularity.png"), (4, "NMI", "ablation_nmi.png")]:
    fig, ax = plt.subplots(figsize=(13, 6))
    w = 0.8 / len(methods); x = np.arange(len(datasets))
    for i, m in enumerate(methods):
        vals = [next(r[idx] for r in rows if r[0] == d and r[1] == m) for d in datasets]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel(label); ax.set_title(f"Ablation: {label} by method")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150); plt.close(fig)
print("\nsaved output/ablation_modularity.png, output/ablation_nmi.png")
