"""
Role-aware adaptive Phase 2 on top of the LPA-MNI+ML multi-level seed.

Phase 1 reuses lpa_mni.multilevel_seed_communities(). Phase 2 is the same
degree-ordered asynchronous label propagation as LPA-MNI, except each node
picks its vote rule from its local role (_adaptive_rule) and only switches
label if the winner clears an acceptance margin over its current label.

Roles / rules (our design, not from a paper):
  hub       degree >= 90th percentile  -> 'inverse_degree' (vote weight 1/k_v,
            so a hub is not dragged around by other hubs)
  boundary  < 50% of neighbors share the node's label -> 'confidence' (vote
            weight = neighbor's own label agreement, so well-anchored
            neighbors count more)
  core      everything else -> 'majority'
"""

import time
from collections import Counter

import networkx as nx
import numpy as np

from .lpa_mni import multilevel_seed_communities, labels_to_communities

NAME = "LPA-MNI+ML-Adaptive"
MAX_ITER = 100


def _node_confidence(G, labels, v):
    """Fraction of v's neighbors carrying v's own label."""
    nbrs = list(G.neighbors(v))
    if not nbrs:
        return 0.0
    return sum(labels[w] == labels[v] for w in nbrs) / len(nbrs)


def _adaptive_rule(G, labels, u, median_degree, hub_cutoff):
    k = G.degree(u)
    if k >= hub_cutoff and k > median_degree:
        return "inverse_degree", "hub"
    if _node_confidence(G, labels, u) < 0.5:
        return "confidence", "boundary"
    return "majority", "core"


def _label_scores(G, labels, u, rule, conf=None):
    scores = {}
    for v in G.neighbors(u):
        if rule == "confidence":
            w = conf[v] + 1e-6
        elif rule == "inverse_degree":
            w = 1.0 / G.degree(v)
        else:
            w = 1.0
        scores[labels[v]] = scores.get(labels[v], 0.0) + w
    return scores


def phase2(G, initial_labels, adaptive=False, acceptance_margin=0.05,
           max_iter=MAX_ITER):
    labels = dict(initial_labels)
    degrees = dict(G.degree())
    vals = list(degrees.values())
    median_degree = float(np.median(vals)) if vals else 0.0
    hub_cutoff = float(np.quantile(vals, 0.90)) if vals else 0.0
    order = sorted(G.nodes(), key=lambda u: (-degrees[u], str(u)))
    role_counts, updates = Counter(), 0
    start = time.perf_counter()
    for _ in range(max_iter):
        changed = False
        for u in order:
            if degrees[u] == 0:
                continue
            if adaptive:
                rule, role = _adaptive_rule(G, labels, u, median_degree, hub_cutoff)
                role_counts[role] += 1
            else:
                rule = "majority"
            conf = ({v: _node_confidence(G, labels, v) for v in G.neighbors(u)}
                    if rule == "confidence" else None)
            scores = _label_scores(G, labels, u, rule, conf)
            best = max(scores.values())
            winners = {l for l, s in scores.items() if abs(s - best) < 1e-12}
            top = max((v for v in G.neighbors(u) if labels[v] in winners),
                      key=lambda v: (degrees[v], str(v)))
            chosen = labels[top]
            if adaptive and chosen != labels[u] and labels[u] in scores:
                total = sum(scores.values()) or 1.0
                if (best - scores[labels[u]]) / total < acceptance_margin:
                    chosen = labels[u]
            if chosen != labels[u]:
                labels[u] = chosen
                changed = True
                updates += 1
        if not changed:
            break
    return labels, time.perf_counter() - start, updates, role_counts


def detect(G, seed=None):
    seed_labels = multilevel_seed_communities(G)
    labels, *_ = phase2(G, seed_labels, adaptive=True)
    return labels_to_communities(labels)
