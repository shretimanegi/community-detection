"""
Louvain (Blondel et al., 2008), via NetworkX's built-in implementation.
Directly maximizes modularity: greedy local moves, then collapse each
community into a super-node and repeat.
"""

import networkx as nx

NAME = "Louvain"


def detect(G, seed=None):
    return [set(c) for c in nx.algorithms.community.louvain_communities(G, seed=seed)]
