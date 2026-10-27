"""
One loader function per dataset. Each returns (G, ground_truth, is_proxy):
  G             -- an undirected networkx.Graph
  ground_truth  -- dict {node: label}, or None if no ground truth exists
  is_proxy      -- True if ground_truth was defined by us from external
                    knowledge rather than shipped with the dataset (see the
                    per-function docstrings for sourcing/caveats)

DATASETS collects them into a single {name: loader_function} dict that
run.py iterates over -- adding a new dataset later means adding one
function here and one line below, nothing else changes.
"""

import os

import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_karate_club():
    """
    Zachary's Karate Club, built into NetworkX. 34 members, edges = observed
    social ties. Ground truth: each member's real post-split faction ("Mr.
    Hi" vs. "Officer"), stored as the "club" node attribute -- a genuine,
    real-world observed split.
    """
    G = nx.karate_club_graph()
    ground_truth = nx.get_node_attributes(G, "club")
    return G, ground_truth, False


def load_les_miserables():
    """
    Les Misérables character co-occurrence network, built into NetworkX. 77
    characters, edges = co-appearance in a chapter (weighted).

    No ground truth ships with this dataset. We define a PROXY ground truth
    ourselves from the novel's plot -- which storyline/social circle each
    character belongs to -- based on our own knowledge of the book, not the
    graph's edges. This is a subjective literary judgment call, not a
    scientific benchmark; treat NMI against it as illustrative only.
    """
    G = nx.les_miserables_graph()

    ground_truth = {
        # Bishop Myriel's circle (Book 1)
        "Myriel": "Myriel circle", "MlleBaptistine": "Myriel circle",
        "MmeMagloire": "Myriel circle", "Napoleon": "Myriel circle",
        "Champtercier": "Myriel circle", "Cravatte": "Myriel circle",
        "Count": "Myriel circle", "CountessDeLo": "Myriel circle",
        "Geborand": "Myriel circle", "OldMan": "Myriel circle",
        # Valjean's world: the law, Javert's pursuit, the trial
        "Valjean": "Valjean/law", "Javert": "Valjean/law",
        "Bamatabois": "Valjean/law", "Fauchelevent": "Valjean/law",
        "Simplice": "Valjean/law", "Gervais": "Valjean/law",
        "Labarre": "Valjean/law", "Isabeau": "Valjean/law",
        "Scaufflaire": "Valjean/law", "Woman1": "Valjean/law",
        "Woman2": "Valjean/law", "Judge": "Valjean/law",
        "Champmathieu": "Valjean/law", "Brevet": "Valjean/law",
        "Chenildieu": "Valjean/law", "Cochepaille": "Valjean/law",
        "Gribier": "Valjean/law", "MmeDeR": "Valjean/law",
        "MotherInnocent": "Valjean/law", "Perpetue": "Valjean/law",
        "Toussaint": "Valjean/law", "Marguerite": "Valjean/law",
        # Fantine & Tholomyes's group of students and grisettes
        "Fantine": "Fantine/Tholomyes", "Tholomyes": "Fantine/Tholomyes",
        "Listolier": "Fantine/Tholomyes", "Fameuil": "Fantine/Tholomyes",
        "Blacheville": "Fantine/Tholomyes", "Dahlia": "Fantine/Tholomyes",
        "Zephine": "Fantine/Tholomyes", "Favourite": "Fantine/Tholomyes",
        # Thenardiers and the Patron-Minette criminal underworld
        "Thenardier": "Thenardier underworld", "MmeThenardier": "Thenardier underworld",
        "Eponine": "Thenardier underworld", "Anzelma": "Thenardier underworld",
        "Babet": "Thenardier underworld", "Claquesous": "Thenardier underworld",
        "Gueulemer": "Thenardier underworld", "Montparnasse": "Thenardier underworld",
        "Brujon": "Thenardier underworld", "Boulatruelle": "Thenardier underworld",
        "Magnon": "Thenardier underworld", "Jondrette": "Thenardier underworld",
        "MmeBurgon": "Thenardier underworld",
        # Amis de l'ABC student revolutionaries, incl. Marius and Gavroche
        "Enjolras": "Amis de l'ABC", "Combeferre": "Amis de l'ABC",
        "Prouvaire": "Amis de l'ABC", "Feuilly": "Amis de l'ABC",
        "Courfeyrac": "Amis de l'ABC", "Bahorel": "Amis de l'ABC",
        "Joly": "Amis de l'ABC", "Bossuet": "Amis de l'ABC",
        "Grantaire": "Amis de l'ABC", "Mabeuf": "Amis de l'ABC",
        "MmeHucheloup": "Amis de l'ABC", "MotherPlutarch": "Amis de l'ABC",
        "Marius": "Amis de l'ABC", "Gavroche": "Amis de l'ABC",
        # Gillenormand household / Cosette & Marius's marriage plot
        "Gillenormand": "Gillenormand household", "MlleGillenormand": "Gillenormand household",
        "LtGillenormand": "Gillenormand household", "MlleVaubois": "Gillenormand household",
        "MmePontmercy": "Gillenormand household", "Pontmercy": "Gillenormand household",
        "BaronessT": "Gillenormand household", "Cosette": "Gillenormand household",
        # Minor children characters
        "Child1": "Children", "Child2": "Children",
    }
    return G, ground_truth, True


def load_florentine_families():
    """
    Florentine Families marriage-ties network, built into NetworkX. 15
    Renaissance-era families, edges = historical marriage ties (Padgett's
    data).

    No ground truth ships with this dataset. We define a PROXY ground truth
    from documented history: the Medici party vs. the Albizzi-led oligarch
    party (Padgett & Ansell, "Robust Action and the Rise of the Medici,
    1400-1434," American Journal of Sociology, 1993) -- the standard
    historical account behind this exact dataset. A few families' allegiance
    (notably Guadagni and Pazzi) is debated among historians, so treat NMI
    against it with lower confidence than Karate Club's real observed split.
    """
    G = nx.florentine_families_graph()
    ground_truth = {
        "Medici": "Medici party", "Ridolfi": "Medici party",
        "Tornabuoni": "Medici party", "Barbadori": "Medici party",
        "Salviati": "Medici party", "Acciaiuoli": "Medici party",
        "Guadagni": "Medici party",
        "Albizzi": "Oligarch party", "Strozzi": "Oligarch party",
        "Peruzzi": "Oligarch party", "Bischeri": "Oligarch party",
        "Castellani": "Oligarch party", "Pazzi": "Oligarch party",
        "Ginori": "Oligarch party", "Lamberteschi": "Oligarch party",
    }
    return G, ground_truth, True


def load_football():
    """
    American College Football network (Girvan & Newman, 2002). 115 Division
    IA teams, edges = regular-season games played in Fall 2000. Ground
    truth: each team's real conference (12 conferences), stored as the
    "value" node attribute -- a genuine, well-established benchmark.

    Loaded as a MultiGraph first because 3 team pairs played twice that
    season (parallel edges in the raw file), then collapsed into a simple
    Graph -- a repeated game shouldn't count as 2 votes for community
    detection purposes.
    """
    path = os.path.join(DATA_DIR, "football.gml")
    MG = nx.read_gml(path)
    G = nx.Graph(MG)
    ground_truth = nx.get_node_attributes(G, "value")
    return G, ground_truth, False


def load_dolphins():
    """
    Dolphin social network (Lusseau et al., 2003; file from Mark Newman's
    network data page). 62 bottlenose dolphins in Doubtful Sound, NZ; edges
    = frequent association. 159 edges, one connected component.

    PROXY ground truth: Newman's GML carries only names, and Lusseau's
    field-observed two-group split is not in the file. The two groups used
    here are the first split of Girvan-Newman edge-betweenness clustering
    (21 vs 41 dolphins, Q = 0.379) -- the two-way "standard division" that
    much of the literature uses in place of the field data. It is derived
    from the graph's own edges, so NMI against it favours structure-based
    algorithms and is circular; treat it as illustrative.
    """
    G = nx.read_gml(os.path.join(DATA_DIR, "dolphins.gml"))
    two = next(nx.algorithms.community.girvan_newman(G))
    ground_truth = {n: i for i, comm in enumerate(two) for n in comm}
    return G, ground_truth, True


def load_cora():
    """
    Cora citation network (standard LINQS release). 2708 machine-learning
    papers, edges = citation links (treated as undirected -- modularity/
    community detection here don't use citation direction as a signal).
    Ground truth: each paper's real research topic (7 classes) -- a genuine
    benchmark label.
    """
    G = nx.Graph()
    ground_truth = {}

    with open(os.path.join(DATA_DIR, "cora.content")) as f:
        for line in f:
            parts = line.strip().split()
            paper_id = parts[0]
            ground_truth[paper_id] = parts[-1]
            G.add_node(paper_id)

    with open(os.path.join(DATA_DIR, "cora.cites")) as f:
        for line in f:
            cited, citing = line.strip().split()
            G.add_edge(cited, citing)

    return G, ground_truth, False


def load_yeast():
    """
    Yeast protein-protein interaction network (Network Repository,
    "bio-yeast"; Rossi & Ahmed, AAAI 2015). 1458 proteins, 1948
    interactions, one connected component, anonymized integer node IDs.

    PROXY ground truth: the file ships no labels and the IDs cannot be
    mapped to proteins, so real functional annotations are unavailable. The
    labels used here are the Girvan-Newman edge-betweenness dendrogram cut
    at its maximum-modularity level (39 communities, Q = 0.814), computed
    with igraph on every load (~16 s). It is derived from the graph's own
    edges, so NMI against it is circular and favours structure-based and
    modularity-driven methods; treat it as illustrative only.
    """
    import igraph as ig

    G = nx.Graph()
    with open(os.path.join(DATA_DIR, "bio-yeast.mtx")) as f:
        lines = [ln for ln in f if not ln.startswith("%")]
    for ln in lines[1:]:                      # first line is the size header
        a, b = ln.split()[:2]
        if a != b:
            G.add_edge(int(a), int(b))

    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in G.edges()])
    clustering = g.community_edge_betweenness(directed=False).as_clustering()
    ground_truth = {nodes[i]: c for i, c in enumerate(clustering.membership)}
    return G, ground_truth, True


DATASETS = {
    "Karate Club": load_karate_club,
    "Dolphins": load_dolphins,
    "Florentine Families": load_florentine_families,
    "American College Football": load_football,
    "Cora": load_cora,
    "Yeast": load_yeast,
}
