"""Graph building functions

This module contains various functions used to create graphs from csv files.
"""

__author__ = 'Henry Robbins'
__all__ = ['distance_matrix', 'create_network', 'grid_instance']


import numpy as np
import pandas as pd
import networkx as nx
from typing import Union


def distance_matrix(nodes:pd.DataFrame,
                    manhattan:bool = False,
                    x_i:str = 'x',
                    y_i:str = 'y',
                    x_j:str = 'x',
                    y_j:str = 'y') -> np.ndarray:
    """Compute the distance matrix between the nodes.

    Returns a distance matrix where entry (i,j) corresponds to the distance
    from the node i to node j.

    Args:
        nodes (pd.DataFrame): Dataframe of nodes with at least (x,y) positions.
        manhattan (bool): {True: manhattan distance, False : euclidian}.
        x_i (str): Column with the x coordinate of node i (Defaults to 'x').
        y_i (str): Column with the y coordinate of node i (Defaults to 'y').
        x_j (str): Column with the x coordinate of node j (Defaults to 'x').
        y_j (str): Column with the y coordinate of node j (Defaults to 'y').

    Returns:
        np.ndarray: Distance matrix between these nodes.
    """
    A = np.array(list(zip(nodes[x_i].tolist(), nodes[y_i].tolist())))
    B = np.array(list(zip(nodes[x_j].tolist(), nodes[y_j].tolist())))
    if manhattan:
        return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1])
    else:
        p1 = np.sum(A**2, axis=1)[:, np.newaxis]
        p2 = np.sum(B**2, axis=1)
        p3 = -2 * np.dot(A,B.T)
        return np.sqrt(p1+p2+p3)


def create_network(nodes:pd.DataFrame,
                   edges:pd.DataFrame = None,
                   directed:bool = False,
                   manhattan:bool = False, **kw
                   ) -> Union[nx.Graph, nx.DiGraph]:
    """Return networkx graph derived from the list of nodes/edges.

    If no edges are given, defaults to generating all edges with
    weight equivalent to the euclidean distance between the nodes.

    Args:
        nodes (pd.DataFrame): Dataframe of nodes with positional columns (x,y).
        edges (pd.DataFrame): Dataframe of edges (u,v) with weight w.
        directed (bool): True iff graph is directed (defaults to False).
        manhattan (bool): {True: manhattan distance, False : euclidian}.

    Returns:
        Union[nx.Graph, nx.DiGraph]: Either an undirected or directed graph.
    """
    graph_type = nx.DiGraph if directed else nx.Graph
    if edges is None:
        A = distance_matrix(nodes, manhattan=manhattan, **kw)
        G = nx.convert_matrix.from_numpy_matrix(A=A, create_using=graph_type)
    else:
        G = nx.convert_matrix.from_pandas_edgelist(df=edges,
                                                   source='u',
                                                   target='v',
                                                   edge_attr='weight',
                                                   create_using=graph_type)
    for attr in nodes.columns:
        nx.set_node_attributes(G, pd.Series(nodes[attr]).to_dict(), attr)

    return G


def grid_instance(n:int, m:int, manhattan:bool = False) -> nx.Graph:
    """Return a graph G representing an n x m set of nodes.

    Args:
        n (int): width of the grid.
        m (int): height of the grid.
        manhattan (bool): {True: manhattan distance, False : euclidian}.

    Returns:
        nx.Graph: Graph of n x m inter-connected nodes.
    """
    nodes = pd.DataFrame()
    for i in range(n):
        for j in range(m):
            nodes = nodes.append({'name': str((i,j)),
                                  'x': i,
                                  'y': j}, ignore_index=True)

    return create_network(nodes, directed=False, manhattan=manhattan)
