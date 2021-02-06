Tutorial
========

First, import the :code:`vinal` package (commonly renamed to :code:`vl`).

.. code-block:: python

   import vinal as vl

   # Calling output_notebook() makes show() display plot in a Jupyter Notebook.
   # Without it, a new tab with the plot will be opened.
   from bokeh.io import output_notebook, show
   output_notebook()  # only include if you want plot inline


All of the algorithm and plotting functions take a NetworkX graph as input. The
:code:`build` module provides a way of constructing NetworkX graphs from
:code:`.csv` files. The standard :code:`nodes.csv` and :code:`edges.csv` files
have the following form:

.. tabularcolumns:: ll

+----------------------+--------------+--------------+
| nodes.csv            | x            | y            |
+----------------------+--------------+--------------+
| 0                    | 0            | 0            |
+----------------------+--------------+--------------+
| 1                    | 1            | 5            |
+----------------------+--------------+--------------+
| 2                    | 7            | 6            |
+----------------------+--------------+--------------+
| 3                    | 8            | 6            |
+----------------------+--------------+--------------+
| 4                    | 4            | 6            |
+----------------------+--------------+--------------+

+----------------------+--------------+--------------+
| edges.csv            | u            | v            |
+----------------------+--------------+--------------+
| 0                    | 0            | 1            |
+----------------------+--------------+--------------+
| 1                    | 1            | 2            |
+----------------------+--------------+--------------+
| 2                    | 4            | 3            |
+----------------------+--------------+--------------+
| 3                    | 3            | 1            |
+----------------------+--------------+--------------+
| 4                    | 2            | 4            |
+----------------------+--------------+--------------+

Use pandas to import these :code:`.csv` files as dataframes.

.. code-block:: python

   import pandas as pd
   nodes = pd.read_csv('nodes.csv', index_col=0)
   edges = pd.read_csv('edges.csv', index_col=0)

We can now use :code:`vl.create_network()` to create a NetworkX graph.

.. code-block:: python

   G = vl.create_network(nodes, edges)


If an edges dataframe is not provided, the graph is assumed to be complete
(every pair of nodes is connected) and the weights of each edge are determined
by the euclidean distance between the pair of nodes.

.. code-block:: python

   G = vl.create_network(nodes)

Here, we give a summary of the various graph algorithms one can run.

.. code-block:: python

   # ----- Shortest path problem -----
   # Returns: List of edges in the shortest path tree

   # s: index of source vertex
   tree = vl.dijkstras(G, s=0)


   # ----- Minimum Spanning Tree (MST) -----
   # Returns: List of edges in the shortest path tree

   # i: index of initial vertex
   mst = vl.prims(G, i=0)
   mst = vl.kruskals(G)
   mst = vl.reverse_kruskals(G)

   # returns the cost of the minimum spanning tree
   vl.spanning_tree_cost(G, mst)


   # ----- Traveling Salesman Problem (TSP) -----
   # Returns: List of node indices in the order they are visited

   # i: index of initial vertex
   tour = vl.random_neighbor(G, i=0)
   tour = vl.nearest_neighbor(G, i=0)
   # intial_tour: initial 2-node tour
   tour = vl.nearest_insertion(G, intial_tour=[0,1,0])
   tour = vl.furthest_insertion(G, intial_tour=[0,4,0])
   # tour: initial tour to improve
   tour = vl.two_opt(G, tour=tour)

   # returns the cost of the tour
   vl.tour_cost(G, tour)

There are four types of plots that vinal can generate:

- Static solution plots
- Non-iteractive algorithm visualization plots
- Interactive create plots
- Interactive algorithm plots

After genrating a solution via one of the algorithms, a static plot of the
solution can be shown. In the following example, a tour is found using nearest
insertion and then plotted.

.. code-block:: python

   tour = vl.nearest_insertion(G, initial_tour=[0,1,0])
   show(vl.tour_plot(G, tour))


If one wishes to see each iteration of the algorithm, a plot with a
:code:`Previous` and :code:`Next` button can be generated. In most cases,
the cost of the solution in each iteration is shown and the text "done." will
appear on the final iteration. In the following example, a tour is found using
random neighbor and then a plot is returned showing each iteration of the
2-OPT tour improvement heuristic.

.. code-block:: python

   tour = vl.random neighbor(G)
   show(vl.tsp_heuristic_plot(G, algorithm='2-OPT', tour=tour))


Tours and spanning trees can also be constructed in a point-and-click fashion.
When creating a tour, click the next node you wish to visit. When creating
a spanning tree, click each edge you want in the tree.

.. code-block:: python

   show(vl.create_spanning_tree_plot(G))
   show(vl.create_tour_plot(G))

Lastly, an interactive version of Dijkstra's algorithm and the MST algorithms
can be plotted. For Dijkstra's algorithm, the user is asked to select the next
node from the frontier set to explore. For the MST algorithms, the user is
asked to select the next edge to be added/removed from the tree. In all cases,
a helpful error message will appear when the user selects incorreclty.

.. code-block:: python

   show(vl.assisted_mst_algorithm_plot(G, algorithm='kruskals'))