.. image:: cornell/logo_red.svg
   :width: 50%
   :alt: Cornell University Seal

|

vinal (VIsualize Network ALgorithms)
====================================

vinal is a Python package for visualizing graph/network algorithms. Currently,
the following algorithms are implemented:


- `Shortest path problem <https://en.wikipedia.org/wiki/Shortest_path_problem>`_
    - `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_
- `Minimum Spanning Tree (MST) <https://en.wikipedia.org/wiki/Minimum_spanning_tree>`_
    - `Prim's Algorithm <https://en.wikipedia.org/wiki/Prim%27s_algorithm>`_
    - `Kruskal's Algorithm <https://en.wikipedia.org/wiki/Kruskal%27s_algorithm>`_
    - `Reverse Kruskal's Algorithm <https://en.wikipedia.org/wiki/Reverse-delete_algorithm>`_
- `Travelling Salesman Problem (TSP) <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_
    - Random neighbor
    - `Nearest neighbor <https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm>`_
    - Nearest insertion
    - Furthest insertion
    - `2-OPT <https://en.wikipedia.org/wiki/2-opt>`_

`NetworkX <https://networkx.org/>`_ graphs can be constructed from a single
:code:`.csv` file of node locations. Alternatively, one can specify edges of the
graph by providing an additional :code:`.csv` file of edges. The package relies
on `bokeh <https://docs.bokeh.org/en/latest/index.html>`_ to generate standalone
HTML files which can be viewed in a Jupyter Notebook inline or in a web browser.

.. toctree::
   :maxdepth: 3

   quickstart_guide/index
   examples/index
   modules