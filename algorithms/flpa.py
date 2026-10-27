"""
FLPA: FIFO-queue-based Label Propagation Algorithm (Kothen-Hill, 2023-style
variant). Classic LPA (see lpa.py) sweeps every node once per full pass, in
a freshly reshuffled order each pass, and stops only once an entire pass
changes nothing. FLPA instead keeps a single FIFO queue of "active" nodes:

  - every node starts in its own singleton community and is enqueued once,
    in random order (`seed` controls this shuffle -- the algorithm's only
    source of randomness).
  - pop the front node, compute the (weighted) label most common among its
    neighbors, breaking ties uniformly at random.
  - if that best label differs from the node's current label, relabel it
    and re-enqueue any neighbor that doesn't already hold that label and
    isn't already waiting in the queue.
  - stop when the queue empties -- i.e. no node has an incentive to change
    given its neighbors' current labels.

This tends to converge in fewer node visits than classic LPA, since only
nodes actually affected by a recent change go back on the queue instead of
re-scanning the whole graph every pass.
"""

from collections import deque
import random

import networkx as nx

NAME = "FLPA"


def flpa(G, seed=None, weight=None):
    rng = random.Random(seed)
    label = {v: i for i, v in enumerate(G.nodes())}

    order = list(G.nodes())
    rng.shuffle(order)
    Q = deque(order)
    in_Q = {v: True for v in G.nodes()}

    while Q:
        i = Q.popleft()
        in_Q[i] = False

        counts = {}
        for j in G.neighbors(i):
            w = G[i][j].get(weight, 1) if weight else 1
            counts[label[j]] = counts.get(label[j], 0) + w
        if not counts:
            continue

        best = max(counts.values())
        c = rng.choice([k for k, v in counts.items() if v == best])

        if label[i] != c:
            label[i] = c
            for j in G.neighbors(i):
                if label[j] != c and not in_Q[j]:
                    Q.append(j)
                    in_Q[j] = True

    return label


def detect(G, seed=42):
    weight = "weight" if nx.get_edge_attributes(G, "weight") else None
    label = flpa(G, seed=seed, weight=weight)

    communities = {}
    for node, lbl in label.items():
        communities.setdefault(lbl, set()).add(node)
    return list(communities.values())
