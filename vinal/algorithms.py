"""Graph algorithm functions

This module contains implementations of various graph algorithms.
"""

__author__ = 'Henry Robbins'
__all__ = ['dijkstras', 'prims', 'kruskals', 'reverse_kruskals',
           'spanning_tree_cost', 'neighbor', 'random_neighbor',
           'nearest_neighbor', 'insertion', 'nearest_insertion',
           'furthest_insertion', 'two_opt', 'tour_cost']


import math
import numpy as np
import pandas as pd
import networkx as nx
from random import randrange
from typing import List, Tuple, Union


# -------------
# SHORTEST PATH
# -------------


def dijkstras(G:nx.Graph,
              s:int = 0,
              iterations:bool = False) -> Union[List, Tuple]:
    """Run Dijkstra's algorithm on graph G from source s.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): Source vertex to run the algorithm from. (Defaults to 0)
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List, Tuple]: Shortest path tree edges or iterations.
    """
    # Helper functions
    def create_table(dist, prev, S):
        """Return table for this iteration."""
        df = pd.DataFrame({'label': dist.copy(), 'prev': prev.copy()})
        df['label'] = df['label'].apply(lambda x: '%.1f' % x)
        # asterisk on label of 'settled nodes'
        marks = pd.Series(['*'*(i in S) for i in range(len(df['label']))])
        df['label'] = df['label'].astype(str) + marks.astype(str)
        df['prev'] = df['prev'].apply(lambda x: '-' if math.isnan(x) else x)
        df = df.T
        df.columns = df.columns.astype(str)
        return df.reset_index()

    def shortest_path_tree(prev):
        """Return the shortest path tree defined by prev."""
        edges = [(k,v) for k,v in prev.items() if not math.isnan(v)]
        return edges

    dist = {i: float('inf') for i in range(len(G))}
    prev = {i: float('nan') for i in range(len(G))}
    dist[s] = 0
    S = []
    F = [s]

    tables = [create_table(dist, prev, S)]
    trees = [shortest_path_tree(prev)]
    marks = [S.copy()]

    while len(F) > 0:
        F.sort(reverse=True, key=lambda x: dist[x])
        f = F.pop()
        S.append(f)
        for w in nx.all_neighbors(G,f):
            if w not in S and w not in F:
                dist[w] = dist[f] + G[f][w]['weight']
                prev[w] = f
                F.append(w)
            else:
                if dist[f] + G[f][w]['weight'] < dist[w]:
                    dist[w] = dist[f] + G[f][w]['weight']
                    prev[w] = f
        tables.append(create_table(dist, prev, S))
        trees.append(shortest_path_tree(prev))
        marks.append(S.copy())
    return (marks, trees, tables) if iterations else shortest_path_tree(prev)


# ---------------------------
# MINIMUM SPANNING TREE (MST)
# ---------------------------


def prims(G:nx.Graph(),
          i:int = 0,
          iterations:bool = False) -> Union[List[Tuple[int]], List]:
    """Run Prim's algorithm on graph G starting from node i.

    Args:
        G (nx.Graph): Networkx graph.
        i (int): Index of the node to start from.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[Tuple[int]], List]: Spanning tree or iterations.
    """
    tree = []
    trees = [[]]
    unvisited = list(range(len(G)))
    unvisited.remove(i)
    visited = [i]
    while len(unvisited) > 0:
        possible = {}
        for u in visited:
            for v in unvisited:
                if G.has_edge(u,v):
                    possible[(u,v)] = G[u][v]['weight']
        u,v = min(possible, key=possible.get)
        unvisited.remove(v)
        visited.append(v)
        tree.append((u,v))
        trees.append(list(tree))
    return trees if iterations else tree


def kruskals(G:nx.Graph,
             iterations:bool = False
             ) -> Union[List[Tuple[int]], List]:
    """Run Kruskal's algorithm on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[Tuple[int]], List]: Spanning tree or iterations.
    """
    edges = nx.get_edge_attributes(G,'weight')
    edges = list(dict(sorted(edges.items(), key=lambda item: item[1])))
    tree = []
    trees = [[]]
    forest = {i:i for i in range(len(G))}
    i = 0
    while len(tree) < len(G) - 1:
        u,v = edges[i]
        x = forest[u]
        y = forest[v]
        if x != y:
            for k in [k for k,v in forest.items() if v == y]:
                forest[k] = x
            tree.append((u,v))
            trees.append(list(tree))
        i += 1
    return trees if iterations else tree


def reverse_kruskals(G:nx.Graph,
                     iterations:bool = False
                     ) -> Union[List[Tuple[int]], List]:
    """Run reverse Kruskal's algorithm on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[Tuple[int]], List]: Spanning tree or iterations.
    """
    edges = nx.get_edge_attributes(G,'weight')
    edges = sorted(edges.items(), key=lambda item: item[1], reverse=True)
    edges = list(dict(edges))
    G_prime = nx.Graph()
    for i in range(len(G)):
        G_prime.add_node(i)
    G_prime.add_edges_from(edges)
    trees = [list(G_prime.edges)]
    i = 0
    while len(G_prime.edges) > len(G) - 1:
        u,v = edges[i]
        G_prime.remove_edge(u,v)
        if not nx.is_connected(G_prime):
            G_prime.add_edge(u,v)
        else:
            trees.append(list(G_prime.edges))
        i += 1
    return trees if iterations else list(G_prime.edges)


def spanning_tree_cost(G:nx.Graph, tree:List[Tuple[int]]) -> float:
    """Return the cost of the given spanning tree.

    Args:
        G (nx.Graph): Networkx graph.
        tree (List[Tuple[int]]): List of edges in the spanning tree.

    Return
        float: sum of the edge weights in the spanning tree.
    """
    return sum([G[u][v]['weight'] for u,v in tree])


# ---------------------------------
# TRAVELLING SALESMAN PROBLEM (TSP)
# ---------------------------------


def neighbor(G:nx.Graph,
             i:int = 0,
             nearest:bool = True,
             iterations:bool = False) -> Union[List[int], List[List[int]]]:
    """Run a neighbor heuristic on G starting at the given initial node.

    Args:
        G (nx.Graph): Networkx graph.
        i (int): Index of the node to start from. (Defaults to 0)
        nearest (bool): Run nearest neighbor if true. Otherwise, run random.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[int], List[List[int]]]: Final tour or iterations.
    """
    unvisited = list(range(len(G)))

    tour = [i]
    unvisited.remove(i)
    tours = [tour.copy()]

    while len(unvisited) > 0:
        if nearest:
            u = tour[-1]
            d = {v: G[u][v]['weight'] for v in range(len(G)) if v in unvisited}
            min_val = min(d.values())
            possible = [k for k, v in d.items() if v == min_val]
            next_node = possible[randrange(len(possible))]
        else:
            next_node = unvisited[randrange(len(unvisited))]
        tour.append(next_node)
        unvisited.remove(next_node)
        tours.append(tour.copy())

    tour.append(i)
    tours.append(tour.copy())

    return tours if iterations else tour


def random_neighbor(G:nx.Graph, i:int = 0) -> List[int]:
    """Run the random neighbor heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        i (int): index of the node to start from. (Defaults to 0)

    Returns:
        List[int]: Tour of the graph G
    """
    return neighbor(G, i=i, nearest=False)


def nearest_neighbor(G:nx.Graph, i:int = 0) -> List[int]:
    """Run the nearest neighbor heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        i (int): index of the node to start from. (Defaults to 0)

    Returns:
        List[int]: Tour of the graph G
    """
    return neighbor(G, i=i, nearest=True)


def insertion(G:nx.Graph,
              initial_tour:List[int] = [0,1,0],
              nearest:bool = True,
              iterations:bool = False) -> Union[List[int], List[List[int]]]:
    """Run an insertion heuristic on G starting with the initial 2-node tour.

    Args:
        G (nx.Graph): Networkx graph.
        initial_tour (List[int]): Initial 2-node tour. (Defaults to [0,1,0])
        nearest (bool): Run nearest insertion if true. Otherwise, run random.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[int], List[List[int]]]: Final tour or iterations.
    """
    A = nx.adjacency_matrix(G).todense()

    unvisited = list(range(len(G)))
    tour = list(initial_tour)
    unvisited.remove(initial_tour[0])
    unvisited.remove(initial_tour[1])
    tours = [tour.copy()]

    while len(unvisited) > 0:
        d = A[:,tour].min(axis=1)
        d = {i: float(d[i]) for i in range(len(d)) if i in unvisited}
        if nearest:
            min_val = min(d.values())
            possible = [k for k, v in d.items() if v == min_val]
        else:
            max_val = max(d.values())
            possible = [k for k, v in d.items() if v == max_val]
        next_node = possible[randrange(len(possible))]

        # insert node into tour at minimum cost
        increase = []
        for i in range(len(tour)-1):
            u, v = tour[i], tour[i+1]
            cost_before = A[u,v]
            cost_after = A[u,next_node] + A[next_node,v]
            increase.append(cost_after - cost_before)
        insert_index = increase.index(min(increase))+1
        tour.insert(insert_index, next_node)
        unvisited.remove(next_node)
        tours.append(tour.copy())

    return tours if iterations else tour


def nearest_insertion(G:nx.Graph,
                      initial_tour:List[int] = [0,1,0]) -> List[int]:
    """Run the nearest insertion heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (List[int]): 2-node tour to start from. (Defaults to [0,1,0])

    Returns:
        List[int]: Tour of the graph G
    """
    return insertion(G, initial_tour=initial_tour, nearest=True)


def furthest_insertion(G:nx.Graph, initial_tour:List[int] = None):
    """Run the furthest insertion heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (List[int]): 2-node tour to start from. (Defaults top)

    Returns:
        List[int]: List[int] of the graph G
    """
    if initial_tour is None:
        initial_tour = [0,len(G)-1,0]
    return insertion(G, initial_tour=initial_tour, nearest=False)


def two_opt(G:nx.Graph,
            tour:List[int],
            iterations:bool = False) -> Union[List[int], Tuple]:
    """Run 2-OPT on the initial tour until no improvement can be made.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List[int]): intial tour to be improved.
        iterations (bool): True iff all iterations should be returned.

    Returns:
        Union[List[int], Tuple]: Improved tour or iterations.
    """
    def two_opt_iteration(tour,G):
        for i in range(len(tour)-1):
            for j in range(i,len(tour)-1):
                u_1, u_2 = tour[i], tour[i+1]
                v_1, v_2 = tour[j], tour[j+1]
                if len(np.unique([u_1, u_2, v_1, v_2])) == 4:
                    after_swap = G[u_1][v_1]['weight'] + G[u_2][v_2]['weight']
                    before_swap = G[u_1][u_2]['weight'] + G[v_1][v_2]['weight']
                    if after_swap < before_swap:
                        swap = tour[i+1:j+1]
                        swap.reverse()
                        tour[i+1:j+1] = swap
                        return True, [u_1, u_2, v_1, v_2]
        return False, None

    tours = [tour.copy()]
    swaps = []
    improved, swapped = two_opt_iteration(tour,G)
    while improved:
        tours.append(tour.copy())
        swaps.append(swapped.copy())
        improved, swapped = two_opt_iteration(tour,G)
    swaps.append(None)
    return (tours, swaps) if iterations else tour


def tour_cost(G:nx.Graph, tour:List[int]) -> float:
    """Return the cost of the tour on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List[int]): List[int] of graph G.

    Returns:
        float: Cost of traversing the entire tour.
    """
    return sum([G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1)])
