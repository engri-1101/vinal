"""
vinal (VIsualize Network ALgorithms)
====================================

vinal is a Python package for visualizing graph/network algorithms. Currently,
the following algorithms are implemented:

- Shortest path problem
    - Dijkstra's algorithm
- Minimum Spanning Tree (MST)
    - Prim's Algorithm
    - Kruskal's Algorithm
    - Reverse Kruskal's Algorithm
- Travelling Salesman Problem (TSP)
    - Random neighbor
    - Nearest neighbor
    - Nearest insertion
    - Furthest insertion
    - 2-OPT

NetworkX graphs can be constructed from a single .csv file of node locations.
Alternatively, one can specify edges of the graph by providing an additional
.csv file of edges. The package relies on bokeh to generate standalone HTML
files which can be viewed in a Jupyter Notebook inline or in a web browser.
"""

__author__ = 'Henry Robbins'

from .build import distance_matrix, create_network, grid_instance
from .algorithms import (dijkstras, prims, kruskals, reverse_kruskals,
                         spanning_tree_cost, neighbor, random_neighbor,
                         nearest_neighbor, insertion, nearest_insertion,
                         furthest_insertion, two_opt, tour_cost)
from .plot import (tour_plot, tree_plot, dijkstras_plot, mst_algorithm_plot,
                   tsp_heuristic_plot, create_tour_plot,
                   create_spanning_tree_plot, assisted_mst_algorithm_plot,
                   assisted_dijkstras_plot)
