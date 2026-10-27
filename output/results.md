# Community Detection Results

Each community/dataset spans two rows -- Modularity then NMI -- one column per algorithm. **bold** marks the best-scoring algorithm for that metric's row (modularity and NMI are judged separately, so a different algorithm can win each).

| Community | Metric | LPA | Louvain | Leiden | Infomap | LPA-MNI | FLPA | LPA-S | LPA-MNI+ML | LPA-MNI+ML-Adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| Karate Club | Modularity | 0.3095 | **0.4449** | **0.4449** | 0.4321 | 0.4036 | 0.4173 | 0.4036 | 0.4345 | 0.3634 |
|  | NMI | 0.3636 | 0.5878 | 0.5878 | 0.5684 | **0.8372** | 0.6835 | **0.8372** | 0.6912 | 0.6085 |
| Dolphins* | Modularity | 0.4986 | 0.5188 | 0.5177 | **0.5203** | 0.5034 | 0.5018 | 0.5108 | 0.5113 | **0.5203** |
|  | NMI | 0.5951 | 0.4838 | 0.5345 | 0.6604 | 0.5955 | 0.6050 | 0.6641 | **0.6673** | 0.6604 |
| Florentine Families* | Modularity | 0.3350 | 0.3975 | **0.3987** | 0.3975 | 0.2887 | 0.3513 | 0.2887 | 0.3912 | 0.3975 |
|  | NMI | 0.1590 | **0.4418** | 0.2184 | 0.2775 | 0.3309 | 0.0950 | 0.3309 | 0.3992 | **0.4418** |
| American College Football | Modularity | 0.5831 | **0.6046** | **0.6046** | 0.6031 | 0.5570 | 0.5544 | 0.5807 | **0.6046** | 0.6043 |
|  | NMI | 0.8697 | 0.8903 | 0.8903 | **0.9114** | 0.8971 | 0.8717 | 0.9102 | 0.8903 | 0.8850 |
| Cora | Modularity | 0.6520 | 0.8157 | **0.8198** | 0.7320 | 0.7259 | 0.6945 | 0.7096 | 0.8173 | 0.8158 |
|  | NMI | 0.4094 | **0.4648** | 0.4612 | 0.4123 | 0.4136 | 0.4007 | 0.3982 | 0.4520 | 0.4460 |
| Yeast* | Modularity | 0.6298 | 0.8201 | **0.8210** | 0.7478 | 0.7266 | 0.6901 | 0.7556 | 0.8184 | 0.8182 |
|  | NMI | 0.7427 | 0.7754 | 0.7756 | **0.7920** | 0.7490 | 0.7579 | 0.7065 | 0.7604 | 0.7694 |

\* No built-in ground truth exists for these datasets. NMI is computed against a proxy ground truth defined from external knowledge, not the graph itself:
- **Florentine Families**: historical Medici party vs. Albizzi-led oligarch party (Padgett & Ansell, 1993) — a documented split, though a few families' allegiance is debated.
- **Dolphins**: the first two-way split of Girvan-Newman edge-betweenness clustering (21 vs 41 dolphins), a common stand-in for Lusseau's field-observed groups, which are not in the data file. It is computed from the graph's own edges, so NMI against it is circular and favours structure-based methods.
- **Yeast**: the Girvan-Newman edge-betweenness dendrogram cut at its maximum-modularity level (39 communities). The data file has anonymized node IDs and no functional labels. The labels are computed from the graph's own edges, so NMI against them is circular and favours modularity-driven methods.

Treat these NMI values as illustrative, with lower confidence than Karate Club, American College Football, and Cora (which all have a real, established ground truth).

Summary bar charts across all runs: `output/modularity_histogram.png`, `output/nmi_histogram.png`.
