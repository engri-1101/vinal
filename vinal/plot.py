"""Plotting functions

This module contains various functions to plot graphs and algorithms.
"""

__author__ = 'Henry Robbins'
__all__ = ['plot_graph', 'plot_create', 'plot_graph_iterations',
           'plot_dijkstras', 'plot_mst_algorithm', 'plot_tsp_heuristic']


import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
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

# --------------------
# JAVASCRIPT CONSTANTS
# --------------------

INCREMENT = """
if ((parseInt(n.text) + 1) < parseInt(k.text)) {
    n.text = (parseInt(n.text) + 1).toString()
}
var iteration = parseInt(n.text)
"""

DECREMENT = """
if ((parseInt(n.text) - 1) >= 0) {
    n.text = (parseInt(n.text) - 1).toString()
}
var iteration = parseInt(n.text)
"""

DONE_UPDATE = """
if (iteration == parseInt(k.text) - 1) {
    done.text = "done."
} else {
    done.text = ""
}
"""

EDGE_SUBSET_UPDATE = """
edge_subset_src.data['xs'] = source.data['edge_xs'][iteration]
edge_subset_src.data['ys'] = source.data['edge_ys'][iteration]
edge_subset_src.change.emit()
"""

COST_UPDATE = """
cost.text = source.data['costs'][iteration].toFixed(1)
"""

TABLE_UPDATE = """
table_src.data = source.data['tables'][iteration]
"""

NODES_UPDATE = """
var in_tree = source.data['nodes'][iteration]

for (let i = 0; i < nodes_src.data['line_color'].length ; i++) {
    if (in_tree.includes(i)) {
        nodes_src.data['fill_color'][i] = '""" + PRIMARY_DARK_COLOR + """'
        nodes_src.data['line_color'][i] = '""" + PRIMARY_DARK_COLOR + """'
    } else {
        nodes_src.data['fill_color'][i] = '""" + PRIMARY_LIGHT_COLOR + """'
        nodes_src.data['line_color'][i] = '""" + PRIMARY_DARK_COLOR + """'
    }
}

nodes_src.change.emit()
"""

SWAPS_UPDATE = """
swaps_src.data['swaps_before_x'] = source.data['swaps_before_x'][iteration]
swaps_src.data['swaps_before_y'] = source.data['swaps_before_y'][iteration]
swaps_src.data['swaps_after_x'] = source.data['swaps_after_x'][iteration]
swaps_src.data['swaps_after_y'] = source.data['swaps_after_y'][iteration]
swaps_src.change.emit()
"""

ON_HOVER = """
source.data['last_index'] = cb_data.index.indices[0]
"""


def _graph_range(x, y):
    """Return graph range containing the given points."""
    min_x, max_x = min(x), max(x)
    x_margin = 0.085*(max_x - min_x)
    min_x, max_x = min_x - x_margin, max_x + x_margin
    min_y, max_y = min(y), max(y)
    y_margin = 0.085*(max_y - min_y)
    min_y, max_y = min_y - y_margin, max_y + y_margin
    return min_x, max_x, min_y, max_y


def _blank_plot(G, plot_width=None, plot_height=None, image=None):
    """Return a blank bokeh plot."""
    if image is not None:
        im = Image.open(image)
        max_x, max_y = im.size
        min_x, min_y = 0 ,0
    else:
        x = nx.get_node_attributes(G,'x').values()
        y = nx.get_node_attributes(G,'y').values()
        min_x, max_x, min_y, max_y = _graph_range(x,y)
    plot = figure(x_range=(min_x, max_x),
                  y_range=(min_y, max_y),
                  title="",
                  plot_width=400 if plot_width is None else plot_width,
                  plot_height=400 if plot_height is None else plot_height)
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


def _set_edge_positions(G):
    """Add edge attribute with positional data."""
    xs = {(u,v): (G.nodes[u]['x'], G.nodes[v]['x']) for u,v in G.edges}
    ys = {(u,v): (G.nodes[u]['y'], G.nodes[v]['y']) for u,v in G.edges}
    nx.set_edge_attributes(G, xs, 'xs')
    nx.set_edge_attributes(G, ys, 'ys')


def _set_graph_colors(G, edges=[], show_all_edges=True):
    """Add node/edge attribute with color data. Highlight edges."""
    for u in G.nodes:
        G.nodes[u]['line_color'] = PRIMARY_DARK_COLOR
        G.nodes[u]['line_width'] = NODE_LINE_WIDTH
        G.nodes[u]['fill_color'] = PRIMARY_LIGHT_COLOR
    for u,v in G.edges:
        G[u][v]['line_color'] = TERTIARY_COLOR
        if show_all_edges:
            G[u][v]['visible'] = True
        else:
            G[u][v]['visible'] = False
    for u,v in edges:
        G[u][v]['line_color'] = TERTIARY_DARK_COLOR
        G[u][v]['visible'] = True


def _graph_sources(G):
    """Return data sources for the graph G."""
    nodes_df = pd.DataFrame([G.nodes[u] for u in sorted(G.nodes())])
    nodes_src = ColumnDataSource(data=nodes_df.to_dict(orient='list'))

    edges_df = pd.DataFrame([G[u][v] for u,v in G.edges()])
    u,v = zip(*[(u,v) for u,v in G.edges])
    edges_df['u'] = u
    edges_df['v'] = v
    edges_src = ColumnDataSource(data=edges_df.to_dict(orient='list'))

    lbl_x = [np.mean(i) for i in nx.get_edge_attributes(G, 'xs').values()]
    lbl_y = [np.mean(i) for i in nx.get_edge_attributes(G, 'ys').values()]
    lbl_txt = list(nx.get_edge_attributes(G,'weight').values())
    labels_src = ColumnDataSource(data={'x': lbl_x,
                                        'y': lbl_y,
                                        'text': lbl_txt})
    return nodes_src, edges_src, labels_src


def _graph_glyphs(plot, nodes_src, edges_src, labels_src,
                  show_edges=True, show_labels=True):
    """Add and return glyphs for nodes and edges"""
    edges_glyph = plot.multi_line(xs='xs', ys='ys',
                                  line_color='line_color',
                                  line_cap='round',
                                  hover_line_color=TERTIARY_DARK_COLOR,
                                  line_width=LINE_WIDTH,
                                  nonselection_line_alpha=1,
                                  visible=show_edges,
                                  alpha='visible',
                                  level='image',
                                  source=edges_src)
    nodes_glyph = plot.circle(x='x', y='y', size=NODE_SIZE, level='glyph',
                              line_color='line_color', fill_color='fill_color',
                              line_width='line_width',
                              nonselection_fill_alpha=1,
                              nonselection_line_alpha=1, source=nodes_src)

    labels = LabelSet(x='x', y='y', text='text', render_mode='canvas',
                      visible=show_labels, source=labels_src)
    plot.add_layout(labels)
    return edges_glyph, nodes_glyph


def plot_graph(G, show_all_edges=True, show_labels=True, edges=[],
               width=None, height=None, image=None):
    """Plot the graph G.

    Args:
        G (nx.Graph): Networkx graph.
        edges (List): Edges to highlight.
    """
    G = G.copy()
    plot = _blank_plot(G, plot_width=width,
                       plot_height=height, image=image)

    _set_edge_positions(G)
    _set_graph_colors(G, edges, show_all_edges=show_all_edges)

    nodes_src, edges_src, labels_src = _graph_sources(G)
    edges_glyph, nodes_glyph = _graph_glyphs(plot, nodes_src, edges_src,
                                             labels_src,
                                             show_labels=show_labels)
    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]))
    grid = gridplot([[plot]],
                    plot_width=width, plot_height=height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_create(G, create, width=None, height=None, image=None):
    """Plot a graph in which you can either create a tour or spanning tree.

    Args:
        G (nx.Graph): Networkx graph.
        create (string): {'tour', 'tree'}
    """
    if create == 'tour':
        show_all_edges = False
        show_labels = False
    elif create == 'tree':
        show_all_edges = True
        show_labels = True

    G = G.copy()
    plot = _blank_plot(G, plot_width=width,
                       plot_height=height, image=image)

    _set_edge_positions(G)
    _set_graph_colors(G, show_all_edges=show_all_edges)

    nodes_src, edges_src, labels_src = _graph_sources(G)

    source = ColumnDataSource(data={'tree_nodes': [],
                                    'tree_edges': [],
                                    'clicked': []})
    tour_src = ColumnDataSource(data={'edges_x': [],
                                      'edges_y': []})
    G_matrix = nx.adjacency_matrix(G).todense().tolist()
    cost_matrix = ColumnDataSource(data={'G': G_matrix})
    cost = Div(text='0.0', width=int(plot.plot_width/3), align='center')
    error_msg = Div(text='', width=int(plot.plot_width/3), align='center')
    clicked = Div(text='[]', width=int(plot.plot_width/3), align='center')
    plot.line(x='edges_x', y='edges_y', line_color=TERTIARY_DARK_COLOR,
              line_cap='round', line_width=LINE_WIDTH, level='image',
              source=tour_src)
    edges_glyph, nodes_glyph = _graph_glyphs(plot, nodes_src, edges_src,
                                             labels_src,
                                             show_labels=show_labels)

    check_done = """
    if (error_msg.text == 'done.') {
        return;
    }
    """

    create_tree_on_click = """
    var i = source.data['last_index']
    var u = edges_src.data['u'][i]
    var v = edges_src.data['v'][i]
    var w = edges_src.data['weight'][i]

    var tree_nodes = source.data['tree_nodes']
    var tree_edges = source.data['tree_edges']

    if (!tree_nodes.includes(u) || !tree_nodes.includes(v)) {
        if (!tree_nodes.includes(v)) {
            nodes_src.data['fill_color'][v] = '""" + PRIMARY_DARK_COLOR + """'
            nodes_src.data['line_color'][v] = '""" + PRIMARY_DARK_COLOR + """'
        }
        if (!tree_nodes.includes(u)) {
            nodes_src.data['fill_color'][u] = '""" + PRIMARY_DARK_COLOR + """'
            nodes_src.data['line_color'][u] = '""" + PRIMARY_DARK_COLOR + """'
        }
        edges_src.data['line_color'][i] = '""" + TERTIARY_DARK_COLOR + """'
        tree_edges.push([u,v])
        var prev_cost = parseFloat(cost.text)
        cost.text = (prev_cost + w).toFixed(1)
        error_msg.text = ''
    } else {
        error_msg.text = 'You have selected an invalid edge.'
    }

    if (tree_nodes.length == nodes_src.data['x'].length) {
        error_msg.text = 'done.'
    }

    clicked.text = '['
    for (let i = 0; i < tree_edges.length; i++) {
        var edge_str = tree_edges[i].join(',')
        clicked.text = clicked.text.concat('(').concat(edge_str).concat(')')
        if (!(i == tree_edges.length - 1)) {
            clicked.text = clicked.text.concat(',')
        }
    }
    clicked.text = clicked.text.concat(']')

    source.change.emit()
    nodes_src.change.emit()
    edges_src.change.emit()
    """

    create_tour_on_click = """
    var v = source.data['last_index']
    var n = nodes_src.data['line_color'].length
    var tour = source.data['clicked']

    if (tour.includes(v)) {
        error_msg.text = 'This node is already in the tour.'
        return;
    } else {
        error_msg.text = ''
    }

    function add_node(v) {
        // add to cost
        if (tour.length > 0) {
            var u = tour[tour.length - 1]
            var prev_cost = parseFloat(cost.text)
            cost.text = (prev_cost + cost_matrix.data['G'][u][v]).toFixed(1)
        }

        // add to tour
        tour.push(v)
        clicked.text = '['.concat(tour.join(',')).concat(']')

        // highlight new node
        nodes_src.data['line_color'][v] = '""" + PRIMARY_DARK_COLOR + """'
        nodes_src.data['fill_color'][v] = '""" + PRIMARY_DARK_COLOR + """'

        // highlight new edge
        tour_src.data['edges_x'].push(nodes_src.data['x'][v])
        tour_src.data['edges_y'].push(nodes_src.data['y'][v])
    }

    add_node(v)
    if (tour.length == n) {
        add_node(tour[0])
        error_msg.text = 'done.'
    }

    source.change.emit()
    tour_src.change.emit()
    nodes_src.change.emit()
    """

    if create == 'tree':
        on_click = check_done + create_tree_on_click
    else:
        on_click = check_done + create_tour_on_click

    renderers = [edges_glyph if create == 'tree' else nodes_glyph]
    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]),
                   HoverTool(tooltips=None,
                             callback=CustomJS(args=dict(source=source),
                                               code=ON_HOVER),
                             renderers=renderers),
                   TapTool(callback=CustomJS(args=dict(source=source,
                                                       edges_src=edges_src,
                                                       nodes_src=nodes_src,
                                                       tour_src=tour_src,
                                                       cost_matrix=cost_matrix,
                                                       cost=cost,
                                                       error_msg=error_msg,
                                                       clicked=clicked),
                                             code=on_click),
                           renderers=renderers))

    # create layout
    grid = gridplot([[plot],
                     [row(cost,error_msg,clicked)]],
                    plot_width=width, plot_height=height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_graph_iterations(G, nodes=None, edges=None, costs=None, tables=None,
                          swaps=None, show_edges=True, show_labels=True,
                          width=None, height=None, image=None):
    """Plot the graph G with iterations of edges, nodes, and tables.

    Args:
        G (nx.Graph): Networkx graph.
        nodes (List): Nodes to highlight at each iteration.
        edges (List): Edges to highlight at each iteration.
        tables (List): Tables at each iteration.
    """
    G = G.copy()
    plot = _blank_plot(G, plot_width=width,
                       plot_height=height, image=image)

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
            xs = [G[u][v]['xs'] for u,v in edges[i]]
            ys = [G[u][v]['ys'] for u,v in edges[i]]
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
    nodes_src, edges_src, labels_src = _graph_sources(G)
    args_dict['nodes_src'] = nodes_src

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
    edges_glyph, nodes_glyph = _graph_glyphs(plot, nodes_src, edges_src,
                                             labels_src,
                                             show_edges=show_edges,
                                             show_labels=show_labels)

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
                          width=width, background='white', index_position=None,
                          editable=False, reorderable=False, sortable=False,
                          selectable=False)
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
    next_btn_code = INCREMENT + DONE_UPDATE
    prev_btn_code = DECREMENT + DONE_UPDATE
    if costs is not None:
        next_btn_code += COST_UPDATE
        prev_btn_code += COST_UPDATE
    if edges is not None:
        next_btn_code += EDGE_SUBSET_UPDATE
        prev_btn_code += EDGE_SUBSET_UPDATE
    if tables is not None:
        next_btn_code += TABLE_UPDATE
        prev_btn_code += TABLE_UPDATE
    if nodes is not None:
        next_btn_code += NODES_UPDATE
        prev_btn_code += NODES_UPDATE
    if swaps is not None:
        next_btn_code += SWAPS_UPDATE
        prev_btn_code += SWAPS_UPDATE

    # buttons
    next_button = Button(label="Next", button_type="primary",
                         width_policy='fit', sizing_mode='scale_width')
    next_button.js_on_click(CustomJS(args=args_dict, code=next_btn_code))
    prev_button = Button(label="Previous", button_type="primary",
                         width_policy='fit', sizing_mode='scale_width')
    prev_button.js_on_click(CustomJS(args=args_dict, code=prev_btn_code))

    plot.add_tools(HoverTool(tooltips=[("Node", "$index")],
                             renderers=[nodes_glyph]))

    # create layout
    layout = [[plot],
              [row(prev_button, next_button,
                   max_width=width, sizing_mode='stretch_both')],
              [row(cost, done) if costs else row(done)]]
    if tables is not None:
        layout.insert(1, [table])

    grid = gridplot(layout,
                    plot_width=plot.plot_width,
                    plot_height=plot.plot_height,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    show(grid)


def plot_dijkstras(G, source=0, width=None, height=None):
    """Plot Dijkstra's algorithm running on G.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): Source vertex to run the algorithm from.
    """
    nodes, edges, tables = dijkstras(G, s=source, iterations=True)
    plot_graph_iterations(G, nodes=nodes, edges=edges, tables=tables,
                          width=width, height=height)


def plot_mst_algorithm(G, alg, i=0, width=None, height=None):
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
    plot_graph_iterations(G, nodes=nodes, edges=edges, costs=costs,
                          width=width, height=height)


def plot_tsp_heuristic(G, alg, initial, width=None, height=None, image=None):
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
    edges = [[(tour[i], tour[i+1]) for i in range(len(tour)-1)] for tour in tours]
    costs = [tour_cost(G, tour) for tour in tours]
    plot_graph_iterations(G, nodes=nodes, edges=edges, costs=costs,
                          swaps=swaps, show_edges=False, show_labels=False,
                          width=width, height=height, image=image)
    return tours[-1]
