.. _ex:

TODO: Dijkstra's plots not displaying correctly.

Examples
========

Furthest Insertion on 7x7 Grid
------------------------------

.. code-block:: python

   G = vl.grid_instance(7,7)
   show(vl.tsp_heuristic_plot(G=G,
                              algorithm='furthest_insertion',
                              initial_tour=[0,48,0]))

.. raw:: html
   :file: FURTHEST_INSERTION_ITER.html

|

2-OPT on 9x9 Grid
-----------------

.. code-block:: python

   G = vl.grid_instance(9,9)
   tour = vl.nearest_neighbor(G, i=0)
   show(vl.tsp_heuristic_plot(G, algorithm='2-OPT', tour=tour))

.. raw:: html
   :file: 2-OPT_GRID.html

|

Prim's on Ithaca
----------------

.. code-block:: python

   edges = pd.read_csv('test_data/tompkins_edges.csv', index_col=0)
   nodes = pd.read_csv('test_data/tompkins_nodes.csv', index_col=0)
   G = vl.create_network(nodes, edges)
   show(vl.mst_algorithm_plot(G, 'prims', i=12, width=600))

.. raw:: html
   :file: PRIMS_ITER.html

|

Dijkstra's on Ithaca
--------------------

.. code-block:: python

   edges = pd.read_csv('test_data/tompkins_edges.csv', index_col=0)
   nodes = pd.read_csv('test_data/tompkins_nodes.csv', index_col=0)
   G = vl.create_network(nodes, edges)
   show(vl.dijkstras_plot(G, s=12, width=700, height=400))

.. raw:: html
   :file: DIJKSTRAS_ITER.html

|

Assisted Prim's Algorithm
-------------------------

.. code-block:: python

   edges = pd.read_csv('test_data/fiber_optic_edges.csv', index_col=0)
   nodes = pd.read_csv('test_data/fiber_optic_nodes.csv', index_col=0)
   G = vl.create_network(nodes, edges)
   show(vl.assisted_mst_algorithm_plot(G, algorithm='prims', s=0, width=700, height=450))

.. raw:: html
   :file: ASSISTED_PRIMS.html

|

Assisted Dijkstra's on Ithaca
-----------------------------

.. code-block:: python

   edges = pd.read_csv('test_data/tompkins_edges.csv', index_col=0)
   nodes = pd.read_csv('test_data/tompkins_nodes.csv', index_col=0)
   G = vl.create_network(nodes, edges)
   show(vl.assisted_dijkstras_plot(G, s=12, width=600))

.. raw:: html
   :file: ASSISTED_DIJKSTRAS.html

|
