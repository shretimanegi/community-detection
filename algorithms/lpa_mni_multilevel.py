"""
LPA-MNI+ML: LPA-MNI (Li et al., Entropy 2021) with Phase 1 replaced by
Louvain-style multi-level coarsening instead of the paper's single flat
local-moving pass -- see algorithms/lpa_mni.py's multilevel_seed_communities()
for the mechanics.

Why: the paper's flat Phase 1 only ever lets a node join a *neighboring
node's* community; it never merges two whole communities together the way
repeated coarsen-and-reoptimize can, so it structurally caps out below
Louvain/Leiden's modularity no matter how it's tuned. Multi-level closes
most of that gap -- e.g. on this project's datasets, Cora's modularity
went from 0.726 (plain LPA-MNI) to 0.818 (within 0.004 of Leiden's 0.822),
and Florentine Families from 0.289 to 0.391 (matching Louvain/Leiden/
Infomap). See algorithms/lpa_mni.py's module docstring and NAME="LPA-MNI"
entry for the unmodified baseline this is compared against.

Cost: a higher-modularity partition can drift from a small ground-truth
clustering (the classic resolution-limit tradeoff) -- Karate Club's NMI
dropped 0.837 -> 0.691 as its community count went from 2 (matching the
real 2-faction split) to 3. Not a bug -- it's the same tradeoff Louvain/
Leiden pay for their modularity lead, on a graph small enough for it to
show up starkly. On the other datasets tested it isn't a straightforward
loss: Florentine Families' NMI *improved* (0.331 -> 0.399) alongside its
modularity, and Football's stayed flat (0.8903, unchanged from plain
LPA-MNI) while its modularity still rose.
"""

from .lpa_mni import run_lpa_mni, labels_to_communities

NAME = "LPA-MNI+ML"


def detect(G, seed=None):
    labels = run_lpa_mni(G, multilevel=True)
    return labels_to_communities(labels)
