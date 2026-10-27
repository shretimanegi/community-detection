"""
Single entry point: runs every algorithm in algorithms/ against every
dataset in datasets.py, scores each partition with modularity and NMI,
saves everything to output/results.csv, then regenerates output/results.md
and the two summary histograms.

Usage:
    ./venv/bin/python run.py
"""

import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from algorithms import ALGORITHMS
from datasets import DATASETS
from results_store import save_result
import generate_table
import plot_results


def compute_nmi(G, communities, ground_truth):
    """None if this dataset has no ground truth; otherwise NMI of the
    detected partition against it."""
    if ground_truth is None:
        return None
    node_to_community = {n: idx for idx, comm in enumerate(communities) for n in comm}
    node_order = sorted(G.nodes())
    predicted = [node_to_community[n] for n in node_order]
    true = [ground_truth[n] for n in node_order]
    return normalized_mutual_info_score(true, predicted)


def main():
    for dataset_name, load in DATASETS.items():
        G, ground_truth, is_proxy = load()
        tag = " (proxy ground truth)" if is_proxy else ""
        print(f"\n=== {dataset_name}{tag} — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ===")

        for algo_name, detect in ALGORITHMS.items():
            communities = detect(G)
            modularity = nx.algorithms.community.modularity(G, communities)
            nmi = compute_nmi(G, communities, ground_truth)
            nmi_str = f"{nmi:.4f}" if nmi is not None else "N/A"

            print(f"  {algo_name:8s}  communities={len(communities):5d}  "
                  f"modularity={modularity:.4f}  nmi={nmi_str}")

            save_result(community=dataset_name, algorithm=algo_name,
                        modularity=modularity, nmi=nmi)

    print()
    generate_table.generate()
    plot_results.generate()


if __name__ == "__main__":
    main()
