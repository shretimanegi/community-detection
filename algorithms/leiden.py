"""
Leiden (Traag, Waltman & van Eck, 2019), via leidenalg on an igraph copy
of G. Like Louvain it maximizes modularity, but its refinement step
guarantees every community is internally connected.
"""

import igraph as ig
import leidenalg

NAME = "Leiden"


def detect(G, seed=None):
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in G.edges()])
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return [{nodes[i] for i in community} for community in part]
