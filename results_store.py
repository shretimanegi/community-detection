"""
Shared results store: run.py calls save_result() once per (dataset,
algorithm) run, appending to output/results.csv. generate_table.py and
plot_results.py both read it back via load_results() to build the table
and the summary histograms -- one source of truth for all three.
"""

import csv
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")

FIELDNAMES = ["community", "algorithm", "modularity", "nmi"]


def save_result(community, algorithm, modularity, nmi):
    """
    Add (or update) one row: a single (community, algorithm) pair's scores.
    Re-running the same pair overwrites its row instead of duplicating it.
    nmi may be None (no ground truth for this dataset) -- stored as "".
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = _read_all()

    key = (community, algorithm)
    replaced = False
    for row in rows:
        if (row["community"], row["algorithm"]) == key:
            row["modularity"] = modularity
            row["nmi"] = nmi if nmi is not None else ""
            replaced = True
            break
    if not replaced:
        rows.append({
            "community": community,
            "algorithm": algorithm,
            "modularity": modularity,
            "nmi": nmi if nmi is not None else "",
        })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_all():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def load_results():
    """
    Returns all recorded rows as a list of dicts with modularity/nmi already
    converted to float (or None where NMI wasn't computable).
    """
    rows = []
    for row in _read_all():
        def to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        rows.append({
            "community": row["community"],
            "algorithm": row["algorithm"],
            "modularity": to_float(row["modularity"]),
            "nmi": to_float(row["nmi"]),
        })
    return rows
