# Community Detection Benchmarks

Benchmarks several community detection algorithms, including the LPA-MNI
family (with a multilevel seed and an adaptive Phase 2), against standard
baselines on small real-world networks. Each partition is scored with
modularity and NMI.

## Algorithms (`algorithms/`)

LPA, FLPA, LPA-S, MFLP, LPA-MNI, LPA-MNI+ML (multilevel), LPA-MNI adaptive,
plus Louvain, Leiden and Infomap as baselines.

## Datasets (`datasets.py`)

Karate Club, Les Misérables, Florentine Families, Football, Dolphins, Cora
and Yeast. Some datasets use a proxy ground truth, marked with `*` in the
results table.

The `data/` and `output/` folders are **not tracked in git**. Place the
following files in `data/` before running:

- `football.gml`, `dolphins.gml`
- `cora.content`, `cora.cites`
- `bio-yeast.mtx`

Karate Club, Les Misérables and Florentine Families ship with NetworkX.

## Setup

```bash
python -m venv venv
./venv/bin/pip install networkx scikit-learn matplotlib numpy
# plus leiden/infomap dependencies used in algorithms/ (e.g. igraph, leidenalg, infomap)
```

## Usage

```bash
./venv/bin/python run.py              # run all algorithms on all datasets
./venv/bin/python ablation_adaptive.py  # ablation of the adaptive Phase 2
./venv/bin/python generate_table.py   # refresh results table only
./venv/bin/python plot_results.py     # refresh plots only
```

Results are written to `output/` (`results.csv`, `results.md`, histograms
and ablation plots).
