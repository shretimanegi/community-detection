"""
One file per algorithm, each exposing:
  NAME             -- display name used in results/tables/plots
  detect(G, seed)  -- returns a list of node sets (communities)

ALGORITHMS collects them into a single {name: detect_function} dict that
run.py iterates over -- adding a new algorithm later means adding one file
here and one line below, nothing else changes.
"""

from . import (lpa, louvain, leiden, infomap_algo, lpa_mni, flpa, lpa_s,
               lpa_mni_multilevel, lpa_mni_adaptive)

ALGORITHMS = {
    lpa.NAME: lpa.detect,
    louvain.NAME: louvain.detect,
    leiden.NAME: leiden.detect,
    infomap_algo.NAME: infomap_algo.detect,
    lpa_mni.NAME: lpa_mni.detect,
    flpa.NAME: flpa.detect,
    lpa_s.NAME: lpa_s.detect,
    lpa_mni_multilevel.NAME: lpa_mni_multilevel.detect,
    lpa_mni_adaptive.NAME: lpa_mni_adaptive.detect,
}
