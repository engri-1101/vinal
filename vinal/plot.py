"""Plotting functions

This module contains various functions to plot graphs and algorithms.
"""

__author__ = 'Henry Robbins'
__all__ = ['plot_graph', 'plot_graph_iterations', 'plot_tour',
           'plot_tree', 'plot_dijkstras', 'plot_mst_algorithm',
           'plot_tsp_heuristic', 'plot_etching_tour', 'plot_create_tour',
           'plot_create_spanning_tree', 'plot_assisted_mst_algorithm']


import numpy as np
import pandas as pd
import networkx as nx
import warnings
from PIL import Image
from pkg_resources import resource_stream
from .algorithms import (dijkstras, prims, kruskals, reverse_kruskals,
                         spanning_tree_cost, neighbor, insertion, two_opt,
                         tour_cost)
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.tables import TableColumn, DataTable
from bokeh.layouts import row, gridplot
from bokeh.models import (HoverTool, TapTool, ColumnDataSource, LabelSet,
                          Button, CustomJS)


# Color Theme -- Using Google's Material Design Color System
# https://material.io/design/color/the-color-system.html

PRIMARY_COLOR = '#1565c0'
PRIMARY_LIGHT_COLOR = '#5e92f3'
PRIMARY_DARK_COLOR = '#003c8f'
SECONDARY_COLOR = '#d50000'
SECONDARY_LIGHT_COLOR = '#ff5131'
SECONDARY_DARK_COLOR = '#9b0000'
PRIMARY_FONT_COLOR = '#ffffff'
SECONDARY_FONT_COLOR = '#ffffff'
# Grayscale
TERTIARY_COLOR = '#DFDFDF'
TERTIARY_LIGHT_COLOR = 'white'  # Jupyter Notebook: white, Sphinx: #FCFCFC
TERTIARY_DARK_COLOR = '#404040'

NODE_SIZE = 11
NODE_LINE_WIDTH = 3
LINE_WIDTH = 5

JS_CODE = resource_stream('vinal.resources', 'plot.js').read().decode("utf-8")


def _graph_range(x, y):
    """Return graph range containing the given points."""
    min_x, max_x = min(x), max(x)
    x_margin = 0.085*(max_x - min_x)
    min_x, max_x = min_x - x_margin, max_x + x_margin
    min_y, max_y = min(y), max(y)
    y_margin = 0.085*(max_y - min_y)
    min_y, max_y = min_y - y_margin, max_y + y_margin
    return min_x, max_x, min_y, max_y


def _blank_plot(G, width=None, height=None, image=None):
    """Return a blank bokeh plot."""
    if image is not None:
        im = Image.open(image)
        max_x, max_y = im.size
        min_x, min_y = 0,0
    else:
        if 'x_start' in list(G.nodes[0]):
            x = (list(nx.get_node_attributes(G,'x_start').values()) +
                 list(nx.get_node_attributes(G,'x_end').values()))
            y = (list(nx.get_node_attributes(G,'y_start').values()) +
                 list(nx.get_node_attributes(G,'y_end').values()))
        else:
            x = nx.get_node_attributes(G,'x').values()
            y = nx.get_node_attributes(G,'y').values()
        min_x, max_x, min_y, max_y = _graph_range(x,y)
    plot = figure(x_range=(min_x, max_x),
                  y_range=(min_y, max_y),
                  title="",
                  plot_width=400 if width is None else width,
                  plot_height=400 if height is None else height)
    plot.toolbar.logo = None
    plot.toolbar_location = None
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.background_fill_color = None
    plot.border_fill_color = None
    plot.outline_line_color = None
    plot.toolbar.active_drag = None
    plot.toolbar.active_scroll = None
    if image is not None:
        _add_image(plot, image)
    return plot


def _add_image(plot, image):
    """Add an image to the background of the plot."""
    plot.image_url(url=[image],
                   x=plot.x_range.start,
                   y=plot.y_range.end,
                   w=plot.x_range.end - plot.x_range.start,
                   h=plot.y_range.end - plot.y_range.start,
                   level='image')


def _edge_positions(G, edges, ret='dict'):
    """Return positional data for the given edges on graph G."""
    xs = {(u,v): (G.nodes[u]['x'], G.nodes[v]['x']) for u,v in edges}
    ys = {(u,v): (G.nodes[u]['y'], G.nodes[v]['y']) for u,v in edges}
    if ret == 'dict':
        return xs, ys
    else:
        return list(xs.values()), list(ys.values())


def _set_edge_positions(G):
    """Add edge attribute with positional data."""
    xs, ys = _edge_positions(G, G.edges)
    nx.set_edge_attributes(G, xs, 'xs')
    nx.set_edge_attributes(G, ys, 'ys')


def _set_graph_colors(G):
    """Add node/edge attribute with color data. Highlight edges."""
    for u in G.nodes:
        G.nodes[u]['line_color'] = PRIMARY_DARK_COLOR
        G.nodes[u]['line_width'] = NODE_LINE_WIDTH
        G.nodes[u]['fill_color'] = PRIMARY_LIGHT_COLOR
    for u,v in G.edges:
        G[u][v]['line_color'] = TERTIARY_COLOR


def _add_nodes(G, plot):
    """Add nodes from G to the plot. Return the data source and glyphs.

    Args:
        G (nx.Graph): Networkx graph.
        plot (figure): Plot to add the nodes to.
    """
    nodes_df = pd.DataFrame([G.nodes[u] for u in sorted(G.nodes())])
    nodes_src = ColumnDataSource(data=nodes_df.to_dict(orient='list'))

    nodes_glyph = plot.circle(x='x', y='y', size=NODE_SIZE, level='glyph',
                              line_color='line_color', fill_color='fill_color',
                              line_width='line_width',
                              nonselection_fill_alpha=1,
                              nonselection_line_alpha=1, source=nodes_src)
    return nodes_src, nodes_glyph


def _add_edges(G, plot, show_labels=True,
               hover_line_color=TERTIARY_DARK_COLOR):
    """Add edges from G to the plot. Return the data source and glyphs.

    Args:
        G (nx.Graph): Networkx graph.
        plot (figure): Plot to add the edges to.
        show_labels (bool): True iff each edge should be labeled.
    """
    edges_df = pd.DataFrame([G[u][v] for u,v in G.edges()])
    u,v = zip(*[(u,v) for u,v in G.edges])
    edges_df['u'] = u
    edges_df['v'] = v
    edges_src = ColumnDataSource(data=edges_df.to_dict(orient='list'))

    edges_glyph = plot.multi_line(xs='xs', ys='ys',
                                  line_color='line_color',
                                  line_cap='round',
                                  hover_line_color=hover_line_color,
                                  line_width=LINE_WIDTH,
                                  nonselection_line_alpha=1,
                                  level='image',
                                  source=edges_src)

    if show_labels:
        lbl_x = [np.mean(i) for i in nx.get_edge_attributes(G, 'xs').values()]
        lbl_y = [np.mean(i) for i in nx.get_edge_attributes(G, 'ys').values()]
        lbl_txt = list(nx.get_edge_attributes(G,'weight').values())
        labels_src = ColumnDataSource(data={'x': lbl_x,
                                            'y': lbl_y,
                                            'text': lbl_txt})

        labels = LabelSet(x='x', y='y', text='text', render_mode='canvas',
                          source=labels_src)
        plot.add_layout(labels)

    return edges_src, edges_glyph


def _get_create_divs(plot, cost_txt='0.0', click_txt='[]'):
    """Return the Divs shown by plots in which a tour or tree is created.

    Args:
        cost_text (str, optional): [description]. Defaults to '0.0'.
        clicked_text (str, optional): [description]. Defaults to '[]'.
        errog_msg_text (str, optional): [description]. Defaults to ''.
    """
    cost = Div(text=cost_txt, width=int(plot.plot_width/3), align='center')
    clicked = Div(text=click_txt, width=int(plot.plot_width/3), align='center')
    error_msg = Div(text='', width=int(plot.plot_width/3), align='center')
    return cost, clicked, error_msg


def _get_blank_src_data(G):
    """Return blank source data for plotting the given graph."""
    return {'visited': [],
            'unvisited': list(range(len(G))),
            'tree_edges': [],
            'edge_ids': [],
            'clicked': []}


def _get_edge_src_maps(G, edges_src):
    """TODO: Need good doc here."""
    edge_ids = np.zeros((len(G), len(G)))
    G_matrix = np.zeros((len(G), len(G)))
    edge_pairs = list(zip(edges_src.data['u'], edges_src.data['v']))
    for i in range(len(edge_pairs)):
        u,v = edge_pairs[i]
        edge_ids[u][v] = i
        edge_ids[v][u] = i
        G_matrix[u][v] = G[u][v]['weight']
        G_matrix[v][u] = G[v][u]['weight']
    return edge_ids.tolist(), G_matrix.tolist()


def _add_tools(plot, on_click, nodes_glyph, renderers, source, edges_src=None,
               nodes_src=None, tour_src=None, cost_matrix=None, cost=None,
               error_msg=None, clicked=None):
    """Add hover and tap tools to the plot."""
    on_hover_code = "source.data['last_index'] = cb_data.index.indices[0]"
    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]),
                   HoverTool(tooltips=None,
                             callback=CustomJS(args=dict(source=source),
                                               code=on_hover_code),
                             renderers=[renderers]),
                   TapTool(callback=CustomJS(args=dict(source=source,
                                                       edges_src=edges_src,
                                                       nodes_src=nodes_src,
                                                       tour_src=tour_src,
                                                       cost_matrix=cost_matrix,
                                                       cost=cost,
                                                       error_msg=error_msg,
                                                       clicked=clicked),
                                             code=on_click),
                           renderers=[renderers]))


def _get_grid(plot, cost, error_msg, clicked):
    """Return gridplot with plot and divs"""
    return gridplot([[plot],
                     [row(cost,error_msg,clicked)]],
                    plot_width=plot.plot_width,
                    plot_height=plot.plot_height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})


def plot_graph(G, show_all_edges=True, show_labels=True, edges=[], cost=None,
               **kw):
    """Plot the graph G.

    Args:
        G (nx.Graph): Networkx graph.
        edges (List): Edges to highlight.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    if show_all_edges:
        edges_src, edges_glyph = _add_edges(G, plot, show_labels=show_labels)

    if len(edges) > 0:
        xs, ys = _edge_positions(G, edges, ret='list')
        plot.multi_line(xs=xs, ys=ys, line_cap='round', line_width=LINE_WIDTH,
                        line_color=TERTIARY_DARK_COLOR, level='image')

    cost_txt = '' if cost is None else ('%.1f' % cost)
    cost = Div(text=cost_txt, width=int(plot.plot_width/2), align='center')

    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]))
    grid = gridplot([[plot],
                     [row(cost)]],
                    plot_width=plot.plot_width, plot_height=plot.plot_height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_tour(G, tour, **kw):
    """Plot the tour on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List): Tour of the graph
    """
    cost = tour_cost(G, tour)
    edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
    plot_graph(G=G, show_all_edges=False, show_labels=False, edges=edges,
               cost=cost, **kw)


def plot_tree(G, tree, show_cost=False, **kw):
    """Plot the tree on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tree (List): List of edges in the tree.
    """
    cost = spanning_tree_cost(G, tree)
    plot_graph(G=G, show_all_edges=True, show_labels=True, edges=tree,
               cost=cost, **kw)


def plot_graph_iterations(G, nodes=None, edges=None, costs=None, tables=None,
                          swaps=None, show_all_edges=True, show_labels=True,
                          **kw):
    """Plot the graph G with iterations of edges, nodes, and tables.

    Args:
        G (nx.Graph): Networkx graph.
        nodes (List): Nodes to highlight at each iteration.
        edges (List): Edges to highlight at each iteration.
        tables (List): Tables at each iteration.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    # build source data dictionary
    k = 0  # number of iterations
    source_data = {}
    if edges is not None:
        k = len(edges)
        edge_xs = []
        edge_ys = []
        for i in range(len(edges)):
            xs, ys = _edge_positions(G, edges[i], ret='list')
            edge_xs.append(xs)
            edge_ys.append(ys)
        source_data['edge_xs'] = edge_xs
        source_data['edge_ys'] = edge_ys
    if nodes is not None:
        k = len(nodes)
        source_data['nodes'] = nodes
    if costs is not None:
        k = len(costs)
        source_data['costs'] = costs
    if tables is not None:
        k = len(tables)
        tables = [table.to_dict(orient='list') for table in tables]
        source_data['tables'] = tables
    if swaps is not None:
        k = len(swaps) + 1
        swaps_before_x = []
        swaps_before_y = []
        swaps_after_x = []
        swaps_after_y = []
        for swap in swaps:
            def get_coord(i):
                return G.nodes()[swap[i]]['x'], G.nodes()[swap[i]]['y']
            (u1x, u1y) = get_coord(0)
            (u2x, u2y) = get_coord(1)
            (v1x, v1y) = get_coord(2)
            (v2x, v2y) = get_coord(3)
            swaps_before_x.append([[u1x, u2x],[v1x, v2x]])
            swaps_before_y.append([[u1y, u2y],[v1y, v2y]])
            swaps_after_x.append([[u1x, v1x],[u2x, v2x]])
            swaps_after_y.append([[u1y, v1y],[u2y, v2y]])
        swaps_before_x.append([[],[]])
        swaps_before_y.append([[],[]])
        swaps_after_x.append([[],[]])
        swaps_after_y.append([[],[]])
        source_data['swaps_before_x'] = swaps_before_x
        source_data['swaps_before_y'] = swaps_before_y
        source_data['swaps_after_x'] = swaps_after_x
        source_data['swaps_after_y'] = swaps_after_y

    # data sources and glyphs
    args_dict = {}
    nodes_src, nodes_glyph = _add_nodes(G, plot)
    args_dict['nodes_src'] = nodes_src
    if show_all_edges:
        edges_src, edges_glyph = _add_edges(G, plot, show_labels=show_labels)

    if nodes is not None:
        for i in range(len(nodes_src.data['line_color'])):
            if i in nodes[0]:
                nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
                nodes_src.data['fill_color'][i] = PRIMARY_DARK_COLOR

    source = ColumnDataSource(data=source_data)
    args_dict['source'] = source

    n = Div(text='0', width=plot.plot_width, align='center')
    k = Div(text=str(k), width=plot.plot_width, align='center')
    done = Div(text='', width=int(plot.plot_width/2), align='center')
    args_dict['n'] = n
    args_dict['k'] = k
    args_dict['done'] = done

    if edges is not None:
        edge_subset_src = ColumnDataSource(data={'xs': edge_xs[0],
                                                 'ys': edge_ys[0]})
        plot.multi_line('xs', 'ys', line_color=TERTIARY_DARK_COLOR,
                        line_width=LINE_WIDTH, level='underlay',
                        line_cap='round', source=edge_subset_src)
        args_dict['edge_subset_src'] = edge_subset_src

    if costs is not None:
        cost = Div(text=str(costs[0]),
                   width=int(plot.plot_width/2),
                   align='center')
        args_dict['cost'] = cost

    if tables is not None:
        table_src = ColumnDataSource(data=tables[0])
        columns = [TableColumn(field='index', title='')]
        for i in range(len(tables[0])-1):
            columns.append(TableColumn(field=str(i), title=str(i)))
        table = DataTable(source=table_src, columns=columns, height=80,
                          width_policy='fit', max_width=plot.plot_width,
                          width=plot.plot_width, background='white',
                          index_position=None, editable=False,
                          reorderable=False, sortable=False, selectable=False)
        args_dict['table_src'] = table_src

    if swaps is not None:
        swaps_src = ColumnDataSource(data={'swaps_before_x': swaps_before_x[0],
                                           'swaps_before_y': swaps_before_y[0],
                                           'swaps_after_x': swaps_after_x[0],
                                           'swaps_after_y': swaps_after_y[0]})
        plot.multi_line(xs='swaps_before_x', ys='swaps_before_y',
                        line_color=SECONDARY_COLOR, line_width=LINE_WIDTH,
                        line_cap='round', level='underlay', source=swaps_src)
        plot.multi_line(xs='swaps_after_x', ys='swaps_after_y',
                        line_color=SECONDARY_COLOR, line_width=LINE_WIDTH,
                        line_cap='round', level='underlay',
                        line_dash=[10,12], source=swaps_src)
        args_dict['swaps_src'] = swaps_src

    # Javascript
    btn_code = JS_CODE
    next_btn_code = btn_code + 'increment_iteration()\ndone_update()\n'
    prev_btn_code = btn_code + 'decrement_iteration()\ndone_update()\n'
    if costs is not None:
        next_btn_code += 'cost_update()\n'
        prev_btn_code += 'cost_update()\n'
    if edges is not None:
        next_btn_code += 'edge_subset_update()\n'
        prev_btn_code += 'edge_subset_update()\n'
    if tables is not None:
        next_btn_code += 'table_update()\n'
        prev_btn_code += 'table_update()\n'
    if nodes is not None:
        next_btn_code += 'nodes_update()\n'
        prev_btn_code += 'nodes_update()\n'
    if swaps is not None:
        next_btn_code += 'swaps_update()\n'
        prev_btn_code += 'swaps_update()\n'

    # buttons
    next_button = Button(label="Next", button_type="primary",
                         max_width=int(plot.plot_width/2),
                         width_policy='fit', sizing_mode='stretch_width')
    next_button.js_on_click(CustomJS(args=args_dict, code=next_btn_code))
    prev_button = Button(label="Previous", button_type="primary",
                         max_width=int(plot.plot_width/2),
                         width_policy='fit', sizing_mode='stretch_width')
    prev_button.js_on_click(CustomJS(args=args_dict, code=prev_btn_code))

    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]))

    # create layout
    layout = [[plot],
              [row(prev_button, next_button,
                   max_width=plot.plot_width, sizing_mode='stretch_both')],
              [row(cost, done) if costs else row(done)]]
    if tables is not None:
        layout.insert(1, [table])

    grid = gridplot(layout,
                    plot_width=plot.plot_width,
                    plot_height=plot.plot_height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_dijkstras(G, source=0, **kw):
    """Plot Dijkstra's algorithm running on G.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): Source vertex to run the algorithm from.
    """
    nodes, edges, tables = dijkstras(G, s=source, iterations=True)
    plot_graph_iterations(G, nodes=nodes, edges=edges, tables=tables, **kw)


def plot_mst_algorithm(G, alg, i=0, **kw):
    """Plot the MST algorithm running on G.

    Args:
        G (nx.Graph): Networkx graph.
        alg (str): {'prims', 'kruskals', 'reverse_kruskals'}
        source (int): Source vertex to run the algorithm from.
    """
    if alg == 'prims':
        edges = prims(G, i=i, iterations=True)
    elif alg == 'kruskals':
        edges = kruskals(G, iterations=True)
    elif alg == 'reverse_kruskals':
        edges = reverse_kruskals(G, iterations=True)
    nodes = []
    for edge in edges:
        nodes.append(list(set([item for sublist in edge for item in sublist])))
    costs = [spanning_tree_cost(G, tree) for tree in edges]
    plot_graph_iterations(G, nodes=nodes, edges=edges, costs=costs, **kw)


def plot_tsp_heuristic(G, alg, initial, **kw):
    """Plot the TSP heuristic running on G.

    Args:
        G (nx.Graph): Networkx graph.
        alg (str): {'random_neighbor', 'nearest_neighbor',
                    'nearest_insertion', 'furthest_insertion', '2-OPT'}
        initial (int): Starting index or tour (depending on alg)
    """
    swaps = None
    if alg == 'random_neighbor':
        tours = neighbor(G, initial=initial, nearest=False, iterations=True)
        if len(tours) > 2:
            del tours[-2]
    elif alg == 'nearest_neighbor':
        tours = neighbor(G, initial=initial, nearest=True, iterations=True)
        if len(tours) > 2:
            del tours[-2]
    elif alg == 'nearest_insertion':
        tours = insertion(G, initial=initial, nearest=True, iterations=True)
    elif alg == 'furthest_insertion':
        tours = insertion(G, initial=initial, nearest=False, iterations=True)
    elif alg == '2-OPT':
        tours, swaps = two_opt(G, tour=initial, iterations=True)
    nodes = tours
    edges = []
    for tour in tours:
        edges.append([(tour[i], tour[i+1]) for i in range(len(tour)-1)])
    costs = [tour_cost(G, tour) for tour in tours]
    plot_graph_iterations(G, nodes=nodes, edges=edges, costs=costs,
                          swaps=swaps, show_all_edges=False, show_labels=False,
                          **kw)
    return tours[-1]


def plot_etching_tour(G, tour, **kw):
    """Plot the tour on the etching problem.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List): Tour of the graph
    """
    # nodes
    x_start = list(nx.get_node_attributes(G,'x_start').values())
    x_end = list(nx.get_node_attributes(G,'x_end').values())
    y_start = list(nx.get_node_attributes(G,'y_start').values())
    y_end = list(nx.get_node_attributes(G,'y_end').values())

    node_xs = [(G.nodes[i]['x_start'], G.nodes[i]['x_end']) for i in G.nodes]
    node_ys = [(G.nodes[i]['y_start'], G.nodes[i]['y_end']) for i in G.nodes]

    # tour edges
    xs = []
    ys = []
    for i in range(len(tour)-1):
        xs.append((G.nodes[tour[i]]['x_end'], G.nodes[tour[i+1]]['x_start']))
        ys.append((G.nodes[tour[i]]['y_end'], G.nodes[tour[i+1]]['y_start']))

    plot = _blank_plot(G, **kw)

    cost_text = '%.1f' % tour_cost(G, tour)
    cost = Div(text=cost_text, width=plot.plot_width, align='center')
    plot.multi_line(xs=node_xs, ys=node_ys, line_color='black', line_width=2)
    plot.multi_line(xs=xs, ys=ys, line_color='black', line_width=2,
                    line_dash='dashed')
    plot.circle(x_start, y_start, size=5, line_color='steelblue',
                fill_color='steelblue')
    plot.circle(x_end, y_end, size=5, line_color='#DC0000',
                fill_color='#DC0000')

    # create layout
    grid = gridplot([[plot],[cost]],
                    plot_width=plot.plot_width, plot_height=plot.plot_height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_create_tour(G, **kw):
    """Plot in which you can create a tour.

    Args:
        G (nx.Graph): Networkx graph.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)
    nodes_src, nodes_glyph = _add_nodes(G, plot)

    # docs indicate that each value should be of the same length but this works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = ColumnDataSource(data=_get_blank_src_data(G))

    G_matrix = nx.adjacency_matrix(G).todense().tolist()
    cost_matrix = ColumnDataSource(data={'G': G_matrix})

    tour_src = ColumnDataSource(data={'edges_x': [],
                                      'edges_y': []})
    plot.line(x='edges_x', y='edges_y', line_color=TERTIARY_DARK_COLOR,
              line_cap='round', line_width=LINE_WIDTH, level='image',
              source=tour_src)

    cost, clicked, error_msg = _get_create_divs(plot)

    code = JS_CODE
    on_click = code + 'check_done()\ncreate_tour_on_click()\n'

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderers=nodes_glyph, source=source, nodes_src=nodes_src,
               tour_src=tour_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    show(_get_grid(plot, cost, error_msg, clicked))


def plot_create_spanning_tree(G, **kw):
    """Plot in which you can create a spanning tree.

    Args:
        G (nx.Graph): Networkx graph.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    edges_src, edges_glyph = _add_edges(G, plot, show_labels=True)

    src_data = _get_blank_src_data(G)
    edge_ids, G_matrix = _get_edge_src_maps(G, edges_src)
    src_data['edge_ids'] = edge_ids
    cost_matrix = ColumnDataSource(data={'G': G_matrix})

    # docs indicate that each value should be of the same length but this works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = ColumnDataSource(data=src_data)

    cost, clicked, error_msg = _get_create_divs(plot)

    code = JS_CODE
    on_click = code + "check_done()\nload_data()\nselect_edge()\ntree_update()"

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderers=edges_glyph, source=source, nodes_src=nodes_src,
               edges_src=edges_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    show(_get_grid(plot, cost, error_msg, clicked))


def plot_assisted_mst_algorithm(G, algorithm, source=None, **kw):
    """Plot in which the user creates an MST using the given algorithm.

    Args:
        G (nx.Graph): Networkx graph.
        algorithm (str): {'prims', 'kruskals', 'reverse_kruskals'}
        source (int): Source vertex to run the algorithm from.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    if source is not None:
        nodes_src.data['fill_color'][source] = PRIMARY_DARK_COLOR
        nodes_src.data['line_color'][source] = PRIMARY_DARK_COLOR
    if algorithm == 'reverse_kruskals':
        hover_color = TERTIARY_COLOR
    else:
        hover_color = TERTIARY_DARK_COLOR
    edges_src, edges_glyph = _add_edges(G, plot, show_labels=True,
                                        hover_line_color=hover_color)

    src_data = _get_blank_src_data(G)
    if algorithm == 'prims':
        unvisited = list(range(len(G)))
        unvisited.remove(source)
        src_data['visited'] = [source]
        src_data['unvisited'] = unvisited
    elif algorithm == 'kruskals':
        edges = nx.get_edge_attributes(G,'weight')
        edges = list(dict(sorted(edges.items(), key=lambda item: item[1])))
        src_data['sorted_edges'] = edges
        src_data['forest'] = list(range(len(G)))
        src_data['index'] = [0]
    elif algorithm == 'reverse_kruskals':
        src_data['visited'] = list(range(len(G)))
        src_data['unvisited'] = []
        src_data['tree_edges'] = list(range(len(G.edges)))
        edges = nx.get_edge_attributes(G,'weight')
        edges = sorted(edges.items(), key=lambda item: item[1], reverse=True)
        edges = list(dict(edges))
        src_data['sorted_edges'] = edges
        src_data['index'] = [0]
        edges_src.data['line_color'] = [TERTIARY_DARK_COLOR]*len(G.edges())
        nodes_src.data['fill_color'] = [PRIMARY_DARK_COLOR]*len(G)

    edge_ids, G_matrix = _get_edge_src_maps(G, edges_src)
    src_data['edge_ids'] = edge_ids
    cost_matrix = ColumnDataSource(data={'G': G_matrix})

    # docs indicate that each value should be of the same length but this works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = ColumnDataSource(data=src_data)

    if algorithm == 'reverse_kruskals':
        cost_text = '%.1f' % spanning_tree_cost(G, G.edges)
        clicked_edges = [str(x).replace(' ', '') for x in list(G.edges)]
        click_txt = '[' + ','.join(clicked_edges) + ']'
        cost, clicked, error_msg = _get_create_divs(plot, cost_txt=cost_text,
                                                    click_txt=click_txt)
    else:
        cost, clicked, error_msg = _get_create_divs(plot)

    code = 'check_done()\nload_data()\n%s()\ntree_update()\n' % (algorithm)
    on_click = JS_CODE + code

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderers=edges_glyph, source=source, nodes_src=nodes_src,
               edges_src=edges_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    show(_get_grid(plot, cost, error_msg, clicked))
