"""
MFLP: Multi-level Fast Label Propagation for overlapping community
detection (J. Intell. Inf. Syst. 64:1255-1279, 2026; s10844-025-00998-6).
Implemented from the "Algorithm MFLP" pseudocode supplied by the user,
which follows the paper's Section 3 and equations 1-8.

  Phase 1  Node marking, repeated until no eligible node remains: a node
           whose remaining (unmarked) degree is 1, or 2 with the two
           neighbors adjacent, is marked and removed.
  Phase 2  On the reduced graph G': NOIM (eq. 1-2), self-influence and
           entropy (eq. 3-4), connection strength CS (eq. 5) and AVG_CS
           (eq. 6). Nodes are grouped by degree (ascending) and sorted by
           NOIM_N ascending within a group -- a fixed, deterministic order.
  Phase 3  Label propagation over that order: seed a new label when no
           neighbor has one, else adopt neighbors' labels with
           CS >= AVG_CS; then spread the node's most recent label to
           first-order neighbors (CS >= AVG_CS) and their common
           neighbors, and once more from each of those (second order).
  Phase 4  Multi-label nodes keep the label with the highest eq. 7 score;
           marked nodes inherit, in reverse marking order, the label of
           their highest-degree labelled neighbor; eq. 8 affiliation
           decides (possibly overlapping) membership.

Notes on spots the pseudocode leaves open
  - Edge weights and self-loops are ignored (unweighted, as in the paper).
  - Eq. 4 is implemented as written: H(v) = -sum_j I_v * ln I_j.
  - Eq. 8's denominator |N(v) & N(u)| can be 0; it is clamped to 1.
  - Ties (argmax) go to the lowest label number. No randomness anywhere.
  - Degrees/neighbors in phases 2-4 are those of the reduced graph G';
    eq. 8 (overlap detection) uses the full graph G.

`detect_overlapping(G)` returns the overlapping communities. `detect(G)`
(used by run.py, whose modularity/NMI need a partition) gives every
overlapping node to ONE of its tied communities -- its eq. 7 label if that
is among them, otherwise the lowest-numbered.
"""

import math

import networkx as nx

NAME = "MFLP"


def _add(labels, v, label):
    """Append to v's ordered multi-label set (most recent = last)."""
    if label not in labels[v]:
        labels[v].append(label)


def detect_overlapping(G):
    nodes = list(G.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    nbr = {v: set(u for u in G.neighbors(v) if u != v) for v in nodes}
    deg_full = {v: len(nbr[v]) for v in nodes}

    # ---- Phase 1: node marking ----
    marked, mark_order = set(), []
    found = True
    while found:
        found = False
        for v in nodes:
            if v in marked:
                continue
            D = nbr[v] - marked
            if len(D) == 1 or (len(D) == 2 and _adjacent(nbr, D)):
                marked.add(v)
                mark_order.append(v)
                found = True

    active = [v for v in nodes if v not in marked]
    active_set = set(active)
    nb = {v: nbr[v] & active_set for v in active}          # neighbors in G'
    deg = {v: len(nb[v]) for v in active}

    # ---- Phase 2: importance, connection strength, ordering ----
    Gp = G.subgraph(active)
    cc = nx.clustering(Gp)
    bc = nx.betweenness_centrality(Gp)
    noim = {v: deg[v] - sum(len(nb[v] & nb[j]) for j in nb[v]) * (1 - cc[v]) + bc[v]
            for v in active}
    lo, hi = (min(noim.values()), max(noim.values())) if active else (0, 0)
    span = (hi - lo) or 1.0
    noim_n = {v: 0.5 + (noim[v] - lo) / span * 0.5 for v in active}

    total = sum(deg.values()) or 1
    inf = {v: deg[v] / total for v in active}
    ent = {v: -sum(inf[v] * math.log(inf[j]) for j in nb[v]) for v in active}
    cs = {i: {j: len(nb[i] & nb[j]) / (ent[i] + ent[j]) for j in nb[i]} for i in active}
    avg = {v: (sum(cs[v].values()) / len(nb[v])) if nb[v] else 0.0 for v in active}

    order = sorted(active, key=lambda v: (deg[v], noim_n[v], pos[v]))

    # ---- Phase 3: label propagation ----
    labels = {v: [] for v in active}
    state = {v: 0 for v in active}
    next_label = 0

    for v in order:
        if state[v] == 1:
            continue
        if not labels[v]:
            if not any(labels[u] for u in nb[v]):
                labels[v].append(next_label)
                next_label += 1
            else:
                for u in sorted(nb[v], key=pos.get):
                    if labels[u] and cs[v][u] >= avg[v]:
                        for l in labels[u]:
                            _add(labels, v, l)
                if not labels[v]:
                    labels[v].append(next_label)
                    next_label += 1
        state[v] = 1
        if avg[v] == 0:
            continue
        ell = labels[v][-1]                                 # primary = most recent

        first = []
        for u in sorted(nb[v], key=pos.get):
            if cs[v][u] >= avg[v]:
                _add(labels, u, ell)
                first.append(u)
                for w in sorted(nb[v] & nb[u], key=pos.get):
                    _add(labels, w, ell)

        for u in first:
            if avg[u] <= 0:
                continue
            for w in sorted(nb[u] - {v}, key=pos.get):
                if cs[u][w] >= avg[u]:
                    _add(labels, w, ell)
                    for x in sorted(nb[u] & nb[w], key=pos.get):
                        _add(labels, x, ell)

    for v in order:
        if not labels[v]:
            labels[v].append(next_label)
            next_label += 1

    # ---- Phase 4: label selection (eq. 7), synchronous ----
    label = {}
    for v in active:
        if len(labels[v]) > 1:
            def score(c, v=v):
                return sum(avg[u] for u in nb[v] if c in labels[u])
            label[v] = max(sorted(labels[v]), key=score)
        else:
            label[v] = labels[v][0]

    for v in reversed(mark_order):
        lab_nbrs = [u for u in nbr[v] if u in label]
        if lab_nbrs:
            u = max(lab_nbrs, key=lambda x: (deg_full[x], -pos[x]))
            label[v] = label[u]
        else:                                               # whole component marked
            label[v] = next_label
            next_label += 1

    # ---- overlap detection (eq. 8), on the full graph ----
    membership = {}
    for v in nodes:
        if not nbr[v]:
            membership[v] = {label[v]}
            continue
        affi = {}
        for u in nbr[v]:
            c = label[u]
            affi[c] = affi.get(c, 0.0) + (1 + deg_full[v]) / max(len(nbr[v] & nbr[u]), 1)
        top = max(affi.values())
        membership[v] = {c for c, a in affi.items() if math.isclose(a, top)}

    detect_overlapping.label = label
    detect_overlapping.membership = membership

    comms = {}
    for v, cset in membership.items():
        for c in cset:
            comms.setdefault(c, set()).add(v)
    return list(comms.values())


def _adjacent(nbr, pair):
    a, b = tuple(pair)
    return b in nbr[a]


def detect(G, seed=None):
    # `seed` unused -- MFLP is deterministic. Kept for the shared signature.
    detect_overlapping(G)
    label, membership = detect_overlapping.label, detect_overlapping.membership
    groups = {}
    for v, cset in membership.items():
        c = label[v] if label[v] in cset else min(cset)
        groups.setdefault(c, set()).add(v)
    return list(groups.values())
