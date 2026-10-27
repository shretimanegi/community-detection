"""
LPA-S: similarity-weighted label propagation (alpha = 0.3), implemented
from the paper's Algorithm LPA-S pseudocode (undirected, unweighted --
edge weights in the datasets are ignored).

  Cent(v)   = sum over neighbors u of 1/d(u)                  (Eq. 10)
  Ac(v, u)  = |N(v) & N(u)|                                   (Eq. 5)
  S(v, u)   = alpha * Cent(u) + (1 - alpha) * Ac(v, u)        (Eq. 12)

Every vertex starts with a unique label. Vertices are visited in a fixed
order (Cent descending, unlike LPA's random order). Each vertex sums S
over its neighbors per label and adopts the label with the largest total;
ties are broken at random (`seed` controls this -- the algorithm's only
randomness). Updates are asynchronous. Stops when a full pass changes
nothing, or after MAX_ITER passes.
"""

import random

NAME = "LPA-S"

ALPHA = 0.3
MAX_ITER = 100


def detect(G, seed=None):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    nbrs = {v: [u for u in G.neighbors(v) if u != v] for v in nodes}
    nbr_set = {v: set(nbrs[v]) for v in nodes}
    deg = {v: len(nbrs[v]) for v in nodes}

    cent = {v: sum(1.0 / deg[u] for u in nbrs[v]) for v in nodes}
    sim = {
        v: {u: ALPHA * cent[u] + (1 - ALPHA) * len(nbr_set[v] & nbr_set[u])
            for u in nbrs[v]}
        for v in nodes
    }

    order = sorted(nodes, key=lambda v: -cent[v])
    label = {v: v for v in nodes}

    for _ in range(MAX_ITER):
        changed = False
        for v in order:
            if not nbrs[v]:
                continue
            score = {}
            for u in nbrs[v]:
                score[label[u]] = score.get(label[u], 0.0) + sim[v][u]
            top = max(score.values())
            best = [l for l, s in score.items() if s == top]
            new = best[0] if len(best) == 1 else rng.choice(best)
            if new != label[v]:
                label[v] = new
                changed = True
        if not changed:
            break

    groups = {}
    for v, l in label.items():
        groups.setdefault(l, set()).add(v)
    return list(groups.values())
