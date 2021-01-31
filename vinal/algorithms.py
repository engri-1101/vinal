"""Graph algorithm functions

This module contains implementations of various graph algorithms.
"""

__author__ = 'Henry Robbins'
__all__ = ['dijkstras', 'prims', 'kruskals', 'reverse_kruskals',
           'spanning_tree_cost', 'neighbor', 'random_neighbor',
           'nearest_neighbor', 'insertion', 'nearest_insertion',
           'furthest_insertion' 'two_opt', 'solve_tsp', 'optimal_tour',
           'tour_cost']


import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from random import randrange
from ortools.constraint_solver import pywrapcp

# -------------
# SHORTEST PATH
# -------------


def dijkstras(G, s=0, iterations=False):
    '''Run Dijkstra's algorithm on graph G from source s.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): Source vertex to run the algorithm from.
        iterations (bool): True iff all iterations should be returned.
    '''
    # Helper functions
    def create_table(dist, prev, S):
        """Return table for this iteration."""
        df = pd.DataFrame({'label': dist.copy(), 'prev': prev.copy()})
        df['label'] = df['label'].apply(lambda x: '%.1f' % x)
        # asterisk on label of 'settled nodes'
        marks = pd.Series(['*'*(i in S) for i in range(len(df['label']))])
        df['label'] = df['label'].astype(str) + marks.astype(str)
        df['prev'] = df['prev'].apply(lambda x: '-' if math.isnan(x) else int(x))
        df = df.T
        df.columns = df.columns.astype(str)
        return df.reset_index()

    def edges_from_prev(prev):
        """Return the edges in the shortest path tree defined by prev."""
        return [(k,v) for k,v in prev.items() if not math.isnan(v)]

    dist = {i: float('inf') for i in range(len(G))}
    prev = {i: float('nan') for i in range(len(G))}
    dist[s] = 0
    S = []
    F = [s]

    tables = [create_table(dist, prev, S)]
    prevs = [edges_from_prev(prev)]
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
        prevs.append(edges_from_prev(prev))
        marks.append(S.copy())
    return (marks, prevs, tables) if iterations else edges_from_prev(prev)


# ---------------------------
# MINIMUM SPANNING TREE (MST)
# ---------------------------


def prims(G, i, iterations=False):
    """Run Prim's algorithm on graph G starting from node i.

    Args:
        G (nx.Graph): Networkx graph.
        i (int): Index of the node to start from.
        iterations (bool): True iff all iterations should be returned.
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


def kruskals(G, iterations=False):
    """Run Kruskal's algorithm on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        iterations (bool): True iff all iterations should be returned.
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


def reverse_kruskals(G, iterations=False):
    """Run reverse Kruskal's algorithm on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        iterations (bool): True iff all iterations should be returned.
    """
    edges = nx.get_edge_attributes(G,'weight')
    edges = list(dict(sorted(edges.items(), key=lambda item: item[1], reverse=True)))
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


def spanning_tree_cost(G, tree):
    """Return the cost of the given spanning tree.

    Args:
        G (nx.Graph): Networkx graph.
        tree (List): List of edges in the spanning tree.
    """
    return sum([G[u][v]['weight'] for u,v in tree])


# ---------------------------------
# TRAVELLING SALESMAN PROBLEM (TSP)
# ---------------------------------


def neighbor(G, initial=0, nearest=True, iterations=False):
    """Run a neighbor heuristic on G starting at the given initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (int): index of the node to start from.
        nearest (bool): run nearest neighbor if true. Otherwise, run random.
        iterations (bool): True iff all iterations should be returned.
    """
    unvisited = list(range(len(G)))  # list of nodes

    # start tour at initial and remove it from unvisited
    tour = [initial]
    unvisited.remove(initial)
    tours = [tour.copy()]

    # choose next node from unvisited
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

    # go back to start
    tour.append(initial)
    tours.append(tour.copy())

    return tours if iterations else tour


def random_neighbor(G, initial=0):
    """Run the random neighbor heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (int): index of the node to start from.
    """
    return neighbor(G, initial=initial, nearest=False)


def nearest_neighbor(G, initial=0):
    """Run the nearest neighbor heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (int): index of the node to start from.
    """
    return neighbor(G, initial=initial, nearest=True)


def insertion(G, initial=[0,1,0], nearest=True, iterations=False):
    """Run an insertion heuristic on G starting with the initial 2-node tour.

    Args:
        G (nx.Graph): Networkx graph.
        intial (List[int]): Initial 2-node tour.
        nearest (bool): Run nearest insertion if true. Otherwise, run random.
        iterations (bool): True iff all iterations should be returned.
    """
    A = nx.adjacency_matrix(G).todense()

    unvisited = list(range(len(G)))  # list of nodes
    tour = list(initial)
    unvisited.remove(initial[0])
    unvisited.remove(initial[1])
    tours = [tour.copy()]

    # choose next node from unvisited
    while len(unvisited) > 0:
        d = A[:,tour[-1]].min(axis=1)
        d = {i : d[i] for i in range(len(d)) if i in unvisited}
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


def nearest_insertion(G, initial=[0,1,0]):
    """Run the nearest insertion heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (int): 2-node tour to start from.
    """
    return insertion(G, initial=initial, nearest=True)


def furthest_insertion(G, initial=None):
    """Run the nearest insertion heuristic on G from the initial node.

    Args:
        G (nx.Graph): Networkx graph.
        intial (int): 2-node tour to start from.
    """
    if initial is None:
        initial = [0,len(G)-1,0]
    return insertion(G, initial=initial, nearest=False)


def two_opt(G, tour, iterations=False):
    """Run 2-OPT on the initial tour until no improvement can be made.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List[int]): intial tour to be improved.
        iterations (bool): True iff all iterations should be returned.
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
    return (tours, swaps) if iterations else tour


def solve_tsp(G):
    """Use OR-Tools to get a tour of the graph G.

    Args:
        G (nx.Graph): Networkx graph.
    """
    # number of locations, number of vehicles, start location
    manager = pywrapcp.RoutingIndexManager(len(G), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return G[from_node][to_node]['weight']*10000

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    solution = routing.Solve()
    return get_routes(solution, routing, manager)[0]


def optimal_tour(name):
    """Return an optimal tour for some instance name."""
    with open('data/optimal_tours.pickle', 'rb') as f:
        optimal_tours = pickle.load(f)
    return optimal_tours[name]


def tour_cost(G, tour):
    """Return the cost of the tour on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List[int]): ordered list of nodes visited on the tour.
    """
    return sum([G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1)])
