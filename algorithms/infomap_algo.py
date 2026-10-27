"""
Infomap (Rosvall & Bergstrom, 2008). Minimizes the description length of
a random walk on the graph rather than maximizing modularity.
"""

from infomap import Infomap

NAME = "Infomap"


def detect(G, seed=None):
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    im = Infomap(silent=True, two_level=True, seed=seed if seed is not None else 123)
    for u, v in G.edges():
        im.add_link(index[u], index[v])
    im.run()
    groups = {}
    for node_id, module in im.get_modules().items():
        groups.setdefault(module, set()).add(nodes[node_id])
    # Isolated nodes have no links, so Infomap never sees them.
    seen = set().union(*groups.values()) if groups else set()
    groups_list = list(groups.values()) + [{v} for v in nodes if v not in seen]
    return groups_list
