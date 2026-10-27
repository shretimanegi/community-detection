"""
LPA-MNI (Li et al., Entropy 2021) — faithful implementation from the
paper's pseudocode. This module's own NAME/detect() entry ("LPA-MNI" in
results.csv/results.md) is the paper's Algorithm 2 exactly, unmodified.

It also holds the shared building blocks (modularity_seed_communities,
label_propagation_mni, multilevel_seed_communities) reused by the sibling
algorithms/lpa_mni_multilevel.py entry ("LPA-MNI+ML"), which swaps this
file's flat single-pass Phase 1 for Louvain-style multi-level coarsening
-- see that file's docstring for why, and multilevel_seed_communities()'s
docstring below for how. Comparing the two side by side in the results
table/histograms shows what multi-level aggregation buys (and costs) over
the paper's original algorithm.

Two phases:
  Phase 1 (modularity_seed_communities): greedy modularity-based seeding,
      visiting nodes in degree-descending order, moving each node into
      whichever neighboring community gives the largest modularity gain
      (the paper's step 4).
  Phase 2 (label_propagation_mni): standard async label propagation seeded
      from Phase 1, ties broken by degree centrality (DC) -- the paper's
      steps 5-8.

Phase 1's local-moving step uses the closed-form Louvain delta-Q formula
(Blondel et al., 2008) rather than recomputing global modularity from
scratch per candidate: moving node i (of weighted degree k_i) out of its
current community and into neighbor community c changes total modularity
by
    ΔQ(i -> c) = k_i,in(c)/m - sigma_tot(c) * k_i / (2 * m**2)
evaluated with i already removed from its own community's sigma_tot. The
best candidate is whichever neighbor community maximizes this over the
community i just left -- provably identical to comparing the full-graph
modularity before/after the move, just O(degree(i)) instead of O(E) per
candidate, so it's fast even on Cora's 2708 nodes/5278 edges.
"""

import networkx as nx

NAME = "LPA-MNI"


def degree_centrality_raw(G):
    """DC(i) = k_i / (n - 1), Eq. 4 in the LPA-MNI paper."""
    n = G.number_of_nodes()
    return {node: G.degree(node) / (n - 1) for node in G.nodes()}


def modularity_seed_communities(G, order, weight=None):
    """
    Phase 1: greedy modularity-based seeding.

    For each node (visited in `order`), test moving it into each neighboring
    community and keep whichever move gives the largest modularity gain
    (via the closed-form delta-Q formula above), or leave it put if no
    neighbor community strictly improves on staying. Repeat over all nodes
    until a full pass makes no more moves.

    Returns: dict {node: community_id}
    """
    m = G.size(weight=weight)
    community = {node: node for node in G.nodes()}
    if m == 0:
        return community

    two_m = 2 * m
    deg = dict(G.degree(weight=weight))
    sigma_tot = dict(deg)  # each node starts as its own singleton community

    improved = True
    while improved:
        improved = False
        for node in order:
            current = community[node]
            k_i = deg[node]

            # Pull node out of its current community before scoring candidates.
            sigma_tot[current] -= k_i

            weight_to = {}
            for nb in G.neighbors(node):
                if nb == node:
                    continue
                w = G[node][nb].get(weight, 1) if weight else 1
                weight_to[community[nb]] = weight_to.get(community[nb], 0) + w

            # Baseline: gain of moving back into the community node just
            # left (i.e. not moving). A neighbor community only wins by
            # strictly beating this.
            best_community = current
            best_gain = weight_to.get(current, 0) / m - (sigma_tot.get(current, 0) * k_i) / (two_m * m)

            for c, w_in in weight_to.items():
                if c == current:
                    continue
                gain = w_in / m - (sigma_tot.get(c, 0) * k_i) / (two_m * m)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_community = c

            community[node] = best_community
            sigma_tot[best_community] = sigma_tot.get(best_community, 0) + k_i
            if best_community != current:
                improved = True

    return community


def _aggregate_graph(G, community, weight):
    """
    Louvain-style coarsening: one node per community in `community`, edges
    = sum of inter-community edge weights, self-loops = sum of
    intra-community edge weights (so the coarse node's weighted degree
    still equals the original community's total degree -- self-loops
    count double towards a node's degree, matching each internal edge's
    original contribution to both its endpoints' degrees).
    """
    H = nx.Graph()
    H.add_nodes_from(set(community.values()))
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1) if weight else 1
        cu, cv = community[u], community[v]
        if H.has_edge(cu, cv):
            H[cu][cv]["weight"] += w
        else:
            H.add_edge(cu, cv, weight=w)
    return H


def multilevel_seed_communities(G):
    """
    Phase 1, multi-level: repeat local-moving (modularity_seed_communities)
    on progressively coarser graphs, Louvain-style, instead of the paper's
    single flat pass.

    Each level: run one local-moving pass, then collapse every community it
    found into a single super-node (edges between super-nodes = summed
    inter-community edge weight, self-loops = summed intra-community
    weight) and repeat on that coarser graph. Stops when a level's pass
    moves nobody (community count == node count, i.e. nothing left to
    merge) or collapses everything into one community. Node visiting order
    at every level is by (unweighted) degree, mirroring the paper's
    degree-centrality ordering -- the delta-Q gain calculation itself still
    uses each level's edge weights for correctness.

    Returns: dict {original_node: final_community_id}
    """
    base_weight = "weight" if nx.get_edge_attributes(G, "weight") else None

    level_graph = G
    level_weight = base_weight
    node_to_original = {n: {n} for n in G.nodes()}

    while True:
        order = sorted(level_graph.nodes(), key=lambda n: level_graph.degree(n), reverse=True)
        community = modularity_seed_communities(level_graph, order, weight=level_weight)

        n_nodes = level_graph.number_of_nodes()
        n_communities = len(set(community.values()))
        if n_communities == n_nodes:
            break  # this level merged nothing -- previous level's result is final

        new_node_to_original = {}
        for node, comm in community.items():
            new_node_to_original.setdefault(comm, set()).update(node_to_original[node])
        node_to_original = new_node_to_original

        if n_communities == 1:
            break  # fully collapsed -- nothing left to gain from another level

        level_graph = _aggregate_graph(level_graph, community, level_weight)
        level_weight = "weight"

    label = {}
    for comm_id, original_nodes in node_to_original.items():
        for node in original_nodes:
            label[node] = comm_id
    return label


def label_propagation_mni(G, seed_communities, order):
    """
    Phase 2: label propagation seeded from Phase 1, ties broken by degree
    centrality (DC).

    Asynchronous updates: a label change is visible immediately to the next
    node processed in the same pass (standard for LPA; avoids the
    oscillation fully-synchronous updates can cause).
    """
    labels = dict(seed_communities)
    dc = degree_centrality_raw(G)

    changed = True
    while changed:
        changed = False
        for node in order:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            tally = {}
            for nb in neighbors:
                tally[labels[nb]] = tally.get(labels[nb], 0) + 1

            max_count = max(tally.values())
            tied_labels = {lab for lab, cnt in tally.items() if cnt == max_count}

            if len(tied_labels) == 1:
                new_label = next(iter(tied_labels))
            else:
                tied_neighbors = [nb for nb in neighbors if labels[nb] in tied_labels]
                best_neighbor = max(tied_neighbors, key=lambda v: dc[v])
                new_label = labels[best_neighbor]

            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

    return labels


def run_lpa_mni(G, multilevel=False):
    """
    End-to-end: Phase 1 seeding + Phase 2 propagation. Returns dict
    {node: label}.

    multilevel=False -> the paper's Algorithm 2 exactly: one flat
        local-moving pass seeds Phase 2 (see algorithms/lpa_mni.py's NAME
        entry, "LPA-MNI").
    multilevel=True -> Louvain-style coarsen-and-repeat seeding instead
        (see algorithms/lpa_mni_multilevel.py's NAME entry, "LPA-MNI+ML");
        trades some of the paper-faithful small-cluster NMI for
        substantially higher modularity, closing most of the gap to
        Louvain/Leiden.
    """
    order = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    if multilevel:
        seed = multilevel_seed_communities(G)
    else:
        weight = "weight" if nx.get_edge_attributes(G, "weight") else None
        seed = modularity_seed_communities(G, order, weight=weight)
    return label_propagation_mni(G, seed, order)


def labels_to_communities(labels):
    """dict {node: label} -> list of sets, for nx.algorithms.community.modularity()."""
    groups = {}
    for node, lab in labels.items():
        groups.setdefault(lab, set()).add(node)
    return list(groups.values())


def detect(G, seed=None):
    # seed is accepted for interface parity with the other algorithms/*.py
    # modules but unused -- both phases are fully deterministic (fixed
    # degree-descending order, deterministic tie-breaks). multilevel=False
    # here -- this is the paper's Algorithm 2 unmodified; see
    # lpa_mni_multilevel.py for the modularity-boosted variant.
    labels = run_lpa_mni(G, multilevel=False)
    return labels_to_communities(labels)


if __name__ == "__main__":
    # Sanity check on Karate Club against ground truth, both variants.
    from networkx.algorithms.community import modularity
    from sklearn.metrics import normalized_mutual_info_score

    G = nx.karate_club_graph()
    ground_truth = [G.nodes[n]["club"] for n in G.nodes()]

    for multilevel in (False, True):
        labels = run_lpa_mni(G, multilevel=multilevel)
        communities = labels_to_communities(labels)
        pred = [labels[n] for n in G.nodes()]
        nmi = normalized_mutual_info_score(ground_truth, pred)
        q = modularity(G, communities)
        variant = "LPA-MNI+ML (multi-level)" if multilevel else "LPA-MNI (original)"
        print(f"{variant:26s} k={len(communities)}  Q={q:.3f}  NMI={nmi:.3f}")
