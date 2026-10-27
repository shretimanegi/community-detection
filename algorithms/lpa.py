"""
Label Propagation (Raghavan, Albert & Kumara, 2007), via NetworkX's
built-in semi-synchronous implementation (Cordasco & Gargano, 2010).

Not modularity-optimizing -- it only propagates local majority-vote labels,
so it's the odd one out next to Louvain and Leiden below (both of which
directly maximize modularity). That's expected: it's the baseline these
scripts started with.
"""

import networkx as nx

NAME = "LPA"


def detect(G, seed=None):
    # NetworkX's label_propagation_communities takes no seed -- it's
    # inherently non-deterministic (randomized tie-breaking each run).
    # `seed` is accepted here only so every algorithms/*.py module shares
    # the same detect(G, seed=...) signature; it's unused for LPA.
    return list(nx.algorithms.community.label_propagation_communities(G))
