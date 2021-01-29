"""Graph building functions

This module contains various functions used to create graphs from csv files.
"""

__author__ = 'Henry Robbins'
__all__ = ['distance_matrix', 'create_network', 'grid_instance']


import numpy as np
import pandas as pd
import networkx as nx


def distance_matrix(nodes, manhattan=True):
    """Compute the distance matrix between the nodes.

    Nodes dataframe can have both start and ending locations by having
    columns x_start, y_start and x_end, y_end

    Args:
        nodes (pd.DataFrame): Dataframe of nodes with at least (x,y) positions.
        manhattan (bool): {True: manhattan distance, False : euclidian}.
    """
    if 'x_start' not in nodes.columns:
        A = np.array(list(zip(nodes['x'].tolist(), nodes['y'].tolist())))
        B = A
    else:
        A = np.array(list(zip(nodes['x_start'].tolist(),
                              nodes['y_start'].tolist())))
        B = np.array(list(zip(nodes['x_end'].tolist(),
                              nodes['y_end'].tolist())))

    if manhattan:
        return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1])
    else:
        p1 = np.sum(A**2, axis=1)[:, np.newaxis]
        p2 = np.sum(B**2, axis=1)
        p3 = -2 * np.dot(A,B.T)
        return np.sqrt(p1+p2+p3)


def create_network(nodes, edges=None, directed=False, manhattan=True):
    """Return networkx graph representing list of nodes/edges.

    If no edges are given, defaults to generating all edges with
    weight equivalent to the euclidean distance between the nodes.

    Args:
        nodes (pd.DataFrame): Dataframe of nodes with positional columns (x,y).
        edges (pd.DataFrame): Dataframe of edges (u,v) with weight w.
        directed (bool): True iff graph is directed (defaults to False).
        manhattan (bool): {True: manhattan distance, False : euclidian}.
    """
    graph_type = nx.DiGraph if directed else nx.Graph
    if edges is None:
        G = nx.convert_matrix.from_numpy_matrix(A=distance_matrix(nodes,
                                                manhattan=manhattan),
                                                create_using=graph_type)
    else:
        G = nx.convert_matrix.from_pandas_edgelist(df=edges,
                                                   source='u',
                                                   target='v',
                                                   edge_attr='weight',
                                                   create_using=graph_type)
    for attr in nodes.columns:
        nx.set_node_attributes(G, pd.Series(nodes[attr]).to_dict(), attr)

    return G


def grid_instance(n, m, manhattan=True):
    """Return a graph G representing an n x m set of nodes.

    Args:
        n (int): width of the grid.
        m (int): height of the grid.
        manhattan (bool): {True: manhattan distance, False : euclidian}.
    """
    nodes = pd.DataFrame()
    for i in range(n):
        for j in range(m):
            nodes = nodes.append({'name': str((i,j)),
                                  'x': i,
                                  'y': j}, ignore_index=True)

    return create_network(nodes, directed=False, manhattan=manhattan)
