# <img alt="vinal" src="docs/branding/vinal_color.png" height="90">

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vinal.svg)](https://pypi.python.org/pypi/vinal/)
[![Documentation Status](https://readthedocs.org/projects/vinal/badge/?version=latest)](https://vinal.readthedocs.io/en/latest/?badge=latest)

vinal is a Python package for visualizing graph/network algorithms. Currently, the following algorithms are implemented:

- [Shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem)
    - [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Minimum Spanning Tree (MST)](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
    - [Prim's Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
    - [Kruskal's Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
    - [Reverse Kruskal's Algorithm](https://en.wikipedia.org/wiki/Reverse-delete_algorithm)
- [Travelling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
    - Random neighbor
    - [Nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm)
    - Nearest insertion
    - Furthest insertion
    - [2-OPT](https://en.wikipedia.org/wiki/2-opt)

[NetworkX](https://networkx.org/) graphs can be constructed from a single ```.csv``` file of node locations. Alternatively, one can specify edges of the graph by providing an additional ```.csv``` file of edges. The package relies on [bokeh](https://docs.bokeh.org/en/latest/index.html) to generate standalone HTML files which can be viewed in a Jupyter Notebook inline or in a web browser.

## Examples

![dijkstras](images/dijkstras.png?raw=true)
![prims](images/prims.png?raw=true)
![us_tour](images/us_tour.png?raw=true)

## Installation

The quickest way to get started is with a pip install.

```bash
pip install vinal
```

## Usage

First, import the ```vinal``` package (commonly renamed to ```vl```).

```python
import vinal as vl

# Calling output_notebook() makes show() display plot in a Jupyter Notebook.
# Without it, a new tab with the plot will be opened.
from bokeh.io import output_notebook, show
output_notebook()  # only include if you want plot inline
```

All of the algorithm and plotting functions take a NetworkX graph as input. The```build``` module provides a way of constructing NetworkX graphs from ```.csv``` files. The standard ```nodes.csv``` and ```edges.csv``` files have the following form:

| nodes.csv | x | y |
| --------- | - | - |
| 0         | 0 | 0 |
| 1         | 1 | 5 |
| 2         | 7 | 6 |
| 3         | 8 | 6 |
| 4         | 4 | 6 |

| edges.csv | u | v |
| --------- | - | - |
| 0         | 0 | 1 |
| 1         | 1 | 2 |
| 2         | 4 | 3 |
| 3         | 3 | 1 |
| 4         | 2 | 4 |

Use pandas to import these ```.csv``` files as dataframes.

```python
import pandas as pd
nodes = pd.read_csv('nodes.csv', index_col=0)
edges = pd.read_csv('edges.csv', index_col=0)
```

We can now use ```vl.create_network()``` to create a NetworkX graph.

```python
G = vl.create_network(nodes, edges)
```

If an edges dataframe is not provided, the graph is assumed to be complete (every pair of nodes is connected) and the weights of each edge are determined by the euclidean distance between the pair of nodes.

```python
G = vl.create_network(nodes)
```

Here, we give a summary of the various graph algorithms one can run.

```python
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
```

There are four types of plots that vinal can generate:

- Static solution plots
- Non-iteractive algorithm visualization plots
- Interactive create plots
- Interactive algorithm plots

After genrating a solution via one of the algorithms, a static plot of the solution can be shown. In the following example, a tour is found using nearest insertion and then plotted.

```python
tour = vl.nearest_insertion(G, initial_tour=[0,1,0])
show(vl.tour_plot(G, tour))
```

![nearest_insertion_plot_tour](images/nearest_insertion_tour_plot.png?raw=true)

If one wishes to see each iteration of the algorithm, a plot with a ```Previous``` and ```Next``` button can be generated. In most cases, the cost of the solution in each iteration is shown and the text "done." will appear on the final iteration. In the following example, a tour is found using random neighbor and then a plot is returned showing each iteration of the 2-OPT tour improvement heuristic.

```python
tour = vl.random neighbor(G)
show(vl.tsp_heuristic_plot(G, algorithm='2-OPT', tour=tour))
```

![2-opt](images/2-opt.png?raw=true)

Tours and spanning trees can also be constructed in a point-and-click fashion. When creating a tour, click the next node you wish to visit. When creating a spanning tree, click each edge you want in the tree.

```python
show(vl.create_spanning_tree_plot(G))
show(vl.create_tour_plot(G))
```

![build_tour](images/build_tour.png?raw=true)

Lastly, an interactive version of Dijkstra's algorithm and the MST algorithms can be plotted. For Dijkstra's algorithm, the user is asked to select the next node from the frontier set to explore. For the MST algorithms, the user is asked to select the next edge to be added/removed from the tree. In all cases, a helpful error message will appear when the user selects incorreclty.

```python
show(vl.assisted_mst_algorithm_plot(G, algorithm='kruskals'))
```

![kruskals_assisted](images/kruskals_assisted.png?raw=true)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />
 This work is licensed under a
 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE)
