"""Plotting functions

This module contains various functions to plot graphs and algorithms.
"""

__author__ = 'Henry Robbins'
__all__ = ['tour_plot', 'tree_plot', 'dijkstras_plot', 'mst_algorithm_plot',
           'tsp_heuristic_plot', 'create_tour_plot',
           'create_spanning_tree_plot', 'assisted_mst_algorithm_plot',
           'assisted_dijkstras_plot']


import numpy as np
import pandas as pd
import networkx as nx
import warnings
from PIL import Image
import pkgutil
from typing import List, Tuple, Dict, Union
from .algorithms import (dijkstras, prims, kruskals, reverse_kruskals,
                         spanning_tree_cost, neighbor, insertion, two_opt,
                         tour_cost)
from bokeh.plotting import figure
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.tables import TableColumn, DataTable
from bokeh.models.renderers import GlyphRenderer
from bokeh.layouts import row, gridplot, GridBox
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
PLOT_MARGIN = 0.085
# Order: image, underlay, glyph, guide, annotation, overlay
LABEL_LEVEL = 'annotation'
LABEL_BACKGROUND_LEVEL = 'guide'
NODE_LEVEL = 'glyph'
EDGE_LEVEL = 'underlay'
IMAGE_LEVEL = 'image'

CONSTANTS_JS = pkgutil.get_data(__name__, "resources/constants.js").decode()

tmp = pkgutil.get_data(__name__, "resources/plot_create.js")
PLOT_CREATE_JS = CONSTANTS_JS + tmp.decode()

tmp = pkgutil.get_data(__name__, "resources/plot_graph_iterations.js")
PLOT_GRAPH_ITERATIONS_JS = CONSTANTS_JS + tmp.decode()


# ------------------------
# Plotting Helper Fuctions
# ------------------------


def _graph_range(x:List[float], y:List[float]) -> List[float]:
    """Return x and y ranges that comfortably contain the given points.

    Args:
        x (List[float]): List of x-coordinates of the points.
        y (List[float]): List of y-coordinates of the points.

    Returns:
        List[float]: x-range (min_x, max_x) and y-range (min_y, max_y).
    """
    min_x, max_x = min(x), max(x)
    x_margin = PLOT_MARGIN * (max_x - min_x)
    min_x, max_x = min_x - x_margin, max_x + x_margin
    min_y, max_y = min(y), max(y)
    y_margin = PLOT_MARGIN * (max_y - min_y)
    min_y, max_y = min_y - y_margin, max_y + y_margin
    return (min_x, max_x, min_y, max_y)


def _blank_plot(G:nx.Graph,
                width:int = None,
                height:int = None,
                x_range:List[float] = None,
                y_range:List[float] = None,
                image:str = None, **kw) -> figure:
    """Return a blank bokeh plot.

    The x and y axis ranges are chosen from x_range and y_range first. If one
    or both are not specified then, it defaults to make all nodes of G visible
    or uses the dimensions of the image if an image was provided.

    Args:
        G (nx.Graph): Graph to be plotted on this blank plot.
        width (int, optional): Plot width. Defaults to None.
        height (int, optional): Plot height. Defaults to None.
        x_range (List[float], optional): Range of x-axis (min_x, max_x).
        y_range (List[float], optional): Range of y-axis (min_y, max_y).
        image (str, optional): Path to image file. Defaults to None.

    Returns:
        figure: Blank bokeh plot.
    """
    if x_range is not None and y_range is not None:
        min_x, max_x = x_range
        min_y, max_y = y_range
    else:
        if image is not None:
            im = Image.open(image)
            max_x, max_y = im.size
            min_x, min_y = 0,0
        else:
            x = nx.get_node_attributes(G,'x').values()
            y = nx.get_node_attributes(G,'y').values()
            min_x, max_x, min_y, max_y = _graph_range(x,y)
    plot = figure(x_range=(min_x, max_x),
                  y_range=(min_y, max_y),
                  title="",
                  width=400 if width is None else width,
                  height=400 if height is None else height)
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


def _add_image(plot:figure, image:str):
    """Add an image to the background of the plot.

    Args:
        plot (figure): Plot to add this image to.
        image (str): Path to image.
    """
    im = Image.open(image).convert('RGBA')
    xdim, ydim = im.size

    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))

    view[:,:,:] = np.flipud(np.asarray(im))

    plot.image_rgba(image=[img],
                    x=plot.x_range.start,
                    y=plot.y_range.start,
                    dw=plot.x_range.end - plot.x_range.start,
                    dh=plot.y_range.end - plot.y_range.start,
                    level="image")


def _edge_positions(G:nx.Graph,
                    edges:List[Tuple[float]],
                    return_type:str = 'List') -> Dict[Tuple[int], float]:
    """Return positional data for the edges of the graph.

    Args:
        G (nx.Graph): Graph (nodes must have pos attributes 'x' and 'y').
        edges (List[Tuple[float]]): Edges to compute positional data for.
        return_type (str): Return type. {'List', 'Dict'}.

    Returns:
        Dict[Tuple[int], float]: Dictionary from edges to positions
    """
    xs = {(u,v): (G.nodes[u]['x'], G.nodes[v]['x']) for u,v in edges}
    ys = {(u,v): (G.nodes[u]['y'], G.nodes[v]['y']) for u,v in edges}
    if return_type == 'Dict':
        return xs, ys
    else:
        return list(xs.values()), list(ys.values())


def _set_edge_positions(G:nx.Graph):
    """Add edge attribute with positional data.

    Args:
        G (nx.Graph): Graph to add positional attributes to.
    """
    xs, ys = _edge_positions(G, G.edges, return_type='Dict')
    nx.set_edge_attributes(G, xs, 'xs')
    nx.set_edge_attributes(G, ys, 'ys')


def _swap_positions(G:nx.Graph, swap:List[int]) -> List[List[float]]:
    """Return positional data for all edges in an edge swap.

    Args:
        G (nx.Graph): Networkx Graph.
        swap (List[int]): (u_1, u_2, v_1, v_2) where (u_1, u_2) and (v_1, v_2)
                          are swapped with (u_1, v_1) and (u_2, v_2).
    Returns:
        List[List[float]]: Positional data for all edges in an edge swap.
    """
    def get_coord(i):
        return G.nodes()[swap[i]]['x'], G.nodes()[swap[i]]['y']

    if swap is None:
        before_x, before_y, after_x, after_y = [[[],[]]] * 4
    else:
        (u1x, u1y) = get_coord(0)
        (u2x, u2y) = get_coord(1)
        (v1x, v1y) = get_coord(2)
        (v2x, v2y) = get_coord(3)
        before_x = [[u1x, u2x],[v1x, v2x]]
        before_y = [[u1y, u2y],[v1y, v2y]]
        after_x = [[u1x, v1x],[u2x, v2x]]
        after_y = [[u1y, v1y],[u2y, v2y]]
    return before_x, before_y, after_x, after_y


def _set_graph_colors(G:nx.Graph):
    """Add node/edge attribute with color data. Highlight edges.

    Args:
        G (nx.Graph): Graph to add color attributes to.
    """
    for u in G.nodes:
        G.nodes[u]['line_color'] = PRIMARY_DARK_COLOR
        G.nodes[u]['line_width'] = NODE_LINE_WIDTH
        G.nodes[u]['fill_color'] = PRIMARY_LIGHT_COLOR
    for u,v in G.edges:
        G[u][v]['line_color'] = TERTIARY_COLOR


def _add_nodes(G:nx.Graph,
               plot:figure) -> Union[ColumnDataSource, GlyphRenderer]:
    """Add nodes from G to the plot.

    Args:
        G (nx.Graph): Networkx graph.
        plot (figure): Plot to add the nodes to.

    Returns:
        Union[ColumnDataSource, GlyphRenderer]: node source and glyphs.
    """
    nodes_df = pd.DataFrame([G.nodes[u] for u in sorted(G.nodes())])
    nodes_src = ColumnDataSource(data=nodes_df.to_dict(orient='list'))

    nodes_glyph = plot.circle(x='x', y='y', size=NODE_SIZE, level=NODE_LEVEL,
                              line_color='line_color', fill_color='fill_color',
                              line_width='line_width',
                              nonselection_fill_alpha=1,
                              nonselection_line_alpha=1, source=nodes_src)

    return nodes_src, nodes_glyph


def _add_edges(G:nx.Graph,
               plot:figure,
               show_labels:bool = True,
               hover_line_color:str = TERTIARY_DARK_COLOR
               ) -> Union[ColumnDataSource, GlyphRenderer]:
    """Add edges from G to the plot.

    Args:
        G (nx.Graph): Networkx graph.
        plot (figure): Plot to add the edges to.
        show_labels (bool): True iff each edge should be labeled.
        hover_line_color (str): Color of the edges when hovering over them.

    Returns:
        Union[ColumnDataSource, GlyphRenderer]: edge source and glyphs.
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
                                  level=EDGE_LEVEL,
                                  source=edges_src)

    if show_labels:
        _add_labels(G, plot)

    return edges_src, edges_glyph


def _add_labels(G:nx.Graph, plot:figure):
    """Add labels from G to the plot.

    Args:
        G (nx.Graph): Networkx graph.
        plot (figure): Plot to add the labels to.
    """
    text = [round(w,2) for w in nx.get_edge_attributes(G,'weight').values()]
    xs = np.array(list(nx.get_edge_attributes(G, 'xs').values()))
    ys = np.array(list(nx.get_edge_attributes(G, 'ys').values()))

    x_range = (plot.x_range.end - plot.x_range.start)
    y_range = (plot.y_range.end - plot.y_range.start)
    margin_shift_x = (x_range / (2 * PLOT_MARGIN + 1)) * PLOT_MARGIN
    margin_shift_y = (y_range / (2 * PLOT_MARGIN + 1)) * PLOT_MARGIN
    x_scale = plot.width / x_range
    y_scale = plot.height / y_range

    xs = (xs + margin_shift_x - plot.x_range.start) * x_scale
    ys = (ys + margin_shift_y - plot.y_range.start) * y_scale

    x = np.mean(xs, axis=1)
    y = np.mean(ys, axis=1)
    centers = np.vstack((x,y)).T
    v = np.vstack((xs[:,1] - xs[:,0], ys[:,1] - ys[:,0])).T
    n = np.divide(v.T, np.linalg.norm(v, axis=1)).T

    lbl_size = np.array([len(str(w)) for w in text]) * 3
    tmp = np.multiply(n, lbl_size[:, np.newaxis])
    blank_start = centers - (n * 3 + tmp)
    blank_end = centers + (n * 3 + tmp)
    blank_xs = np.vstack((blank_start[:,0], blank_end[:,0])).T
    blank_ys = np.vstack((blank_start[:,1], blank_end[:,1])).T

    x = (x / x_scale) - margin_shift_x + plot.x_range.start
    y = (y / y_scale) - margin_shift_y + plot.y_range.start
    blank_xs = (blank_xs / x_scale) - margin_shift_x + plot.x_range.start
    blank_ys = (blank_ys / y_scale) - margin_shift_y + plot.y_range.start

    plot.multi_line(xs=blank_xs.tolist(), ys=blank_ys.tolist(),
                    line_color='white', line_width=LINE_WIDTH+1,
                    nonselection_line_alpha=1, level=LABEL_BACKGROUND_LEVEL)

    labels_src = ColumnDataSource(data={'x': x, 'y': y, 'text': text})
    labels = LabelSet(x='x', y='y', text='text', text_align='center',
                      text_baseline='middle', text_font_size='13px',
                      text_color='black', level=LABEL_LEVEL, source=labels_src)
    plot.add_layout(labels)


def _get_create_divs(plot:figure,
                     cost_txt:str = '0.0',
                     click_txt:str = '[]') -> Tuple[Div]:
    """Return the Divs shown by plots in which a tour or tree is created.

    Args:
        plot (figure): The plot to which these Divs are added.
        cost_text (str, optional): Current cost of solution. Defaults to '0.0'.
        clicked_text (str, optional): Clicked objects. Defaults to '[]'.
    """
    cost = Div(text=cost_txt, width=int(plot.width/3), align='center')
    clicked = Div(text=click_txt, width=int(plot.width/3), align='center')
    error_msg = Div(text='', width=int(plot.width/3), align='center')
    return (cost, clicked, error_msg)


def _get_blank_src_data(G:nx.Graph) -> Dict[str, List[int]]:
    """Return default source data for plotting the given graph.

    Args:
        G (nx.Graph): Graph that is being plotted.

    Returns:
        Dict[str, List[int]]: Default source data.
    """
    return {'visited': [],
            'unvisited': list(range(len(G))),
            'tree_edges': [],
            'edge_ids': [],
            'clicked': []}


def _edge_src_maps(G:nx.Graph,
                   edges_src:ColumnDataSource) -> Tuple[List[List[float]]]:
    """Return JS friendly maps from edges to edge_ids and weights.

    Args:
        G (nx.Graph): Graph containing these edges.
        edges_src (ColumnDataSource): Data source for edges.

    Returns:
        Tuple[List[List[float]]]: edge_ids map and weights map.
    """
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


def _add_tools(plot:figure,
               on_click:str,
               nodes_glyph:GlyphRenderer,
               renderer:GlyphRenderer, **kw):
    """Add hover and tap tools to the plot.

    Args:
        plot (figure): Plot to add these tools to.
        on_click (str): JS code to be run when the renderer is clicked.
        nodes_glyph (GlyphRenderer): Glyph for the nodes on this plot.
        renderer (GlyphRenderer): Renderer that should call on_click code.
    """
    on_hover_code = "source.data['last_index'] = cb_data.index.indices[0]"
    plot.add_tools(HoverTool(tooltips=[("Index", "$index"),
                                       ("Name", "@name")],
                             renderers=[nodes_glyph]),
                   HoverTool(tooltips=None,
                             callback=CustomJS(args=dict(source=kw['source']),
                                               code=on_hover_code),
                             renderers=[renderer]),
                   TapTool(callback=CustomJS(args=kw,
                                             code=on_click),
                           renderers=[renderer]))


def _get_grid(plot:figure,
              cost:Div,
              error_msg:Div,
              clicked:Div) -> GridBox:
    """Return a grid containing the given plot and divs

    Args:
        plot (figure): The plot to be in the grid.
        cost_text (Div): Current cost of solution.
        error_msg (Div): Error message.
        clicked_text (Div): Clicked objects.

    Returns:
        GridBox: grid containing the given plot and divs.
    """
    return gridplot([[plot],
                     [row(cost,error_msg,clicked)]],
                    toolbar_location=None,
                    toolbar_options={'logo': None})


# ------------------------
# Static Plotting Fuctions
# ------------------------


def _graph_plot(G:nx.Graph,
                edges:List[Tuple[int]],
                cost:float = None,
                show_all_edges:bool = True,
                show_labels:bool = True, **kw) -> GridBox:
    """Return a plot of the graph G with given edges highlighted.

    Args:
        G (nx.Graph): Graph to be plotted.
        edges (List[Tuple[int]]): Edges to be highlighted.
        cost (float, optional): Cost to be displayed. Defaults to None.
        show_all_edges (bool, optional): True iff all edges should be shown.
        show_labels (bool, optional): True iff labels should be shown.

    Returns:
        GridBox: Plot of the graph G with given edges highlighted.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    if show_all_edges:
        edges_src, edges_glyph = _add_edges(G, plot, show_labels=show_labels)

    if len(edges) > 0:
        xs, ys = _edge_positions(G, edges)
        plot.multi_line(xs=xs, ys=ys, line_cap='round', line_width=LINE_WIDTH,
                        line_color=TERTIARY_DARK_COLOR, level=EDGE_LEVEL)

    cost_txt = '' if cost is None else ('%.1f' % cost)
    cost = Div(text=cost_txt, width=int(plot.width/2), align='center')

    plot.add_tools(HoverTool(tooltips=[("Index", "$index"),
                                       ("Name", "@name")],
                             renderers=[nodes_glyph]))
    grid = gridplot([[plot],
                     [row(cost)]],
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    return grid


def tour_plot(G:nx.Graph, tour:List[int], **kw) -> GridBox:
    """Return a plot of the tour on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tour (List[int]): Tour of the graph.

    Returns:
        GridBox: Plot of the tour on graph G.
    """
    cost = tour_cost(G, tour)
    edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
    return _graph_plot(G=G, show_all_edges=False, show_labels=False,
                       edges=edges, cost=cost, **kw)


def tree_plot(G:nx.Graph,
              tree:List[Tuple[int]],
              show_cost:bool = False, **kw):
    """Return a plot of the tree on graph G.

    Args:
        G (nx.Graph): Networkx graph.
        tree (List[int]): List of edges in the tree.
        show_cost (bool): True if cost of tree should be shown on plot.

    Returns:
        GridBox: Plot of the tree on graph G.
    """
    cost = spanning_tree_cost(G, tree) if show_cost else None
    return _graph_plot(G=G, show_all_edges=True, show_labels=True,
                       edges=tree, cost=cost, **kw)

# ----------------------------
# Non-Static Plotting Fuctions
# ----------------------------


def _graph_iterations_plot(G:nx.Graph,
                           nodes:List[List[int]] = None,
                           edges:List[List[List[int]]] = None,
                           costs:List[float] = None,
                           tables:List[pd.DataFrame] = None,
                           swaps:List[List[int]] = None,
                           show_all_edges:bool = True,
                           show_labels:bool = True, **kw) -> GridBox:
    """Return a plot of multiple iterations of a graph.

    Args:
        G (nx.Graph): Networkx graph.
        nodes (List[List[int]]): Highlighted nodes at every iteration.
        edges (List[List[List[int]]]): Highlighted edges at every iteration.
        costs (List[float]): Cost at every iteration.
        tables (List[pd.DataFrame]): Table to be shown at every iteration.
        swaps (List[List[int]]): Edge swap to highlight at every iteration.
        show_all_edges (bool, optional): True iff all edges should be shown.
        show_labels (bool, optional): True iff labels should be shown.

    Returns:
        GridBox: Plot of multiple iterations of a graph.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    args_dict = {}  # keep track of objects to pass to JS callback

    # nodes and edges
    nodes_src, nodes_glyph = _add_nodes(G, plot)
    args_dict['nodes_src'] = nodes_src
    if nodes is not None:
        for i in nodes[0]:
            nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
            nodes_src.data['fill_color'][i] = PRIMARY_DARK_COLOR

    if show_all_edges:
        edges_src, edges_glyph = _add_edges(G, plot, show_labels=show_labels)

    # current iteration
    n = Div(text='0', width=plot.width, align='center')
    args_dict['n'] = n

    # total number of iterations
    features = [edges, nodes, costs, tables, swaps]
    k = max([0 if feature is None else len(feature) for feature in features])
    k = Div(text=str(k), width=plot.width, align='center')
    args_dict['k'] = k

    # indicate if on final iteration
    done = Div(text='', width=int(plot.width/2), align='center')
    args_dict['done'] = done

    source_data = {}

    if edges is not None:
        tmp = list(zip(*[_edge_positions(G, edge) for edge in edges]))
        edge_xs, edge_ys = tmp
        source_data['edge_xs'] = edge_xs
        source_data['edge_ys'] = edge_ys
        edge_subset_src = ColumnDataSource(data={'xs': edge_xs[0],
                                                 'ys': edge_ys[0]})
        plot.multi_line('xs', 'ys', line_color=TERTIARY_DARK_COLOR,
                        line_width=LINE_WIDTH, level=EDGE_LEVEL,
                        line_cap='round', source=edge_subset_src)
        args_dict['edge_subset_src'] = edge_subset_src

    if nodes is not None:
        source_data['nodes'] = nodes

    if costs is not None:
        source_data['costs'] = costs
        cost = Div(text=str(costs[0]),
                   width=int(plot.width/2),
                   align='center')
        args_dict['cost'] = cost

    if tables is not None:
        tables = [table.to_dict(orient='list') for table in tables]
        source_data['tables'] = tables
        table_src = ColumnDataSource(data=tables[0])
        columns = [TableColumn(field='index', title='')]
        for i in range(len(tables[0])-1):
            columns.append(TableColumn(field=str(i), title=str(i)))
        table = DataTable(source=table_src, columns=columns, height=80,
                          background='white', index_position=None,
                          editable=False, reorderable=False, sortable=False,
                          selectable=False, width=plot.width)
        args_dict['table_src'] = table_src

    if swaps is not None:
        tmp = list(zip(*[_swap_positions(G, swap) for swap in swaps]))
        swaps_before_x, swaps_before_y, swaps_after_x, swaps_after_y = tmp
        source_data['swaps_before_x'] = swaps_before_x
        source_data['swaps_before_y'] = swaps_before_y
        source_data['swaps_after_x'] = swaps_after_x
        source_data['swaps_after_y'] = swaps_after_y
        swaps_src = ColumnDataSource(data={'swaps_before_x': swaps_before_x[0],
                                           'swaps_before_y': swaps_before_y[0],
                                           'swaps_after_x': swaps_after_x[0],
                                           'swaps_after_y': swaps_after_y[0]})
        plot.multi_line(xs='swaps_before_x', ys='swaps_before_y',
                        line_color=SECONDARY_COLOR, line_width=LINE_WIDTH,
                        line_cap='round', level=EDGE_LEVEL, source=swaps_src)
        plot.multi_line(xs='swaps_after_x', ys='swaps_after_y',
                        line_color=SECONDARY_COLOR, line_width=LINE_WIDTH,
                        line_cap='round', level=EDGE_LEVEL,
                        line_dash='dashed', source=swaps_src)
        args_dict['swaps_src'] = swaps_src

    source = ColumnDataSource(data=source_data)
    args_dict['source'] = source

    code = ('done_update()\n'
            + 'cost_update()\n'*(costs is not None)
            + 'edge_subset_update()\n'*(edges is not None)
            + 'table_update()\n'*(tables is not None)
            + 'nodes_update()\n'*(nodes is not None)
            + 'swaps_update()\n'*(swaps is not None))
    next_btn_code = PLOT_GRAPH_ITERATIONS_JS + 'increment_iteration()\n' + code
    prev_btn_code = PLOT_GRAPH_ITERATIONS_JS + 'decrement_iteration()\n' + code

    # buttons
    next_button = Button(label="Next", button_type="primary",
                         max_width=int(plot.width/2),
                         width_policy='fit', sizing_mode='stretch_width')
    next_button.js_on_click(CustomJS(args=args_dict, code=next_btn_code))
    prev_button = Button(label="Previous", button_type="primary",
                         max_width=int(plot.width/2),
                         width_policy='fit', sizing_mode='stretch_width')
    prev_button.js_on_click(CustomJS(args=args_dict, code=prev_btn_code))

    plot.add_tools(HoverTool(tooltips=[("Index", "$index"),
                                       ("Name", "@name")],
                             renderers=[nodes_glyph]))

    # create layout
    layout = [[plot],
              [row(prev_button, next_button,
                   max_width=plot.width, sizing_mode='stretch_both')],
              [row(cost, done) if costs else row(done)]]
    if tables is not None:
        layout.insert(1, [table])

    grid = gridplot(layout,
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    return grid


def dijkstras_plot(G:nx.Graph, s:int = 0, **kw) -> GridBox:
    """Return plot of Dijkstra's algorithm running on graph G with source s.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): Source vertex to run the algorithm from. (Defaults to 0)

    Returns:
        GridBox: Plot of Dijkstra's algorithm running on graph G with source s.
    """
    nodes, trees, tables = dijkstras(G, s=s, iterations=True)
    return _graph_iterations_plot(G, nodes=nodes, edges=trees,
                                  tables=tables, **kw)


def mst_algorithm_plot(G:nx.Graph, algorithm:str, **kw) -> GridBox:
    """Return plot of the given MST algorithm running on the graph G.

    Args:
        G (nx.Graph): Networkx graph.
        algorithm (str): {'prims', 'kruskals', 'reverse_kruskals'}

    Returns:
        GridBox: Plot of the given MST algorithm running on the graph G.
    """
    if algorithm == 'prims':
        trees = prims(G, i=kw['i'], iterations=True)
    elif algorithm == 'kruskals':
        trees = kruskals(G, iterations=True)
    elif algorithm == 'reverse_kruskals':
        trees = reverse_kruskals(G, iterations=True)
    nodes = []
    for tree in trees:
        nodes.append(list(set([item for sublist in tree for item in sublist])))
    costs = [spanning_tree_cost(G, tree) for tree in trees]
    return _graph_iterations_plot(G, nodes=nodes, edges=trees,
                                  costs=costs, **kw)


def tsp_heuristic_plot(G:nx.Graph, algorithm:str, **kw) -> GridBox:
    """Return a plot of the given TSP heuristic running on G.

    Args:
        G (nx.Graph): Networkx graph.
        algorithm (str): {'random_neighbor', 'nearest_neighbor',
                          'nearest_insertion', 'furthest_insertion', '2-OPT'}

    Returns:
        GridBox: Plot of the given TSP heuristic running on G.
    """
    swaps = None
    if algorithm == 'random_neighbor':
        tours = neighbor(G, i=kw['i'],
                         nearest=False, iterations=True)
        if len(tours) > 2:
            del tours[-2]
    elif algorithm == 'nearest_neighbor':
        tours = neighbor(G, i=kw['i'],
                         nearest=True, iterations=True)
        if len(tours) > 2:
            del tours[-2]
    elif algorithm == 'nearest_insertion':
        tours = insertion(G, initial_tour=kw['initial_tour'],
                          nearest=True, iterations=True)
    elif algorithm == 'furthest_insertion':
        tours = insertion(G, initial_tour=kw['initial_tour'],
                          nearest=False, iterations=True)
    elif algorithm == '2-OPT':
        tours, swaps = two_opt(G, tour=kw['tour'], iterations=True)
    nodes = tours
    edges = []
    for tour in tours:
        edges.append([(tour[i], tour[i+1]) for i in range(len(tour)-1)])
    costs = [tour_cost(G, tour) for tour in tours]
    return _graph_iterations_plot(G, nodes=nodes, edges=edges, costs=costs,
                                  swaps=swaps, show_all_edges=False,
                                  show_labels=False, **kw)


# -----------------------------
# Interactive Plotting Fuctions
# -----------------------------


def create_tour_plot(G:nx.Graph, **kw) -> GridBox:
    """Return a plot in which you can create a tour.

    Args:
        G (nx.Graph): Networkx graph.

    Rerturns:
        GridBox: Plot in which you can create a tour.
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
              line_cap='round', line_width=LINE_WIDTH, level=EDGE_LEVEL,
              source=tour_src)

    cost, clicked, error_msg = _get_create_divs(plot)

    on_click = PLOT_CREATE_JS + 'check_done()\ncreate_tour_on_click()\n'

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderer=nodes_glyph, source=source, nodes_src=nodes_src,
               tour_src=tour_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    return _get_grid(plot, cost, error_msg, clicked)


def create_spanning_tree_plot(G:nx.Graph, **kw) -> GridBox:
    """Return a plot in which you can create a spanning tree.

    Args:
        G (nx.Graph): Networkx graph.

    Returns:
        GridBox: Plot in which you can create a spanning tree.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    edges_src, edges_glyph = _add_edges(G, plot, show_labels=True)

    src_data = _get_blank_src_data(G)
    edge_ids, G_matrix = _edge_src_maps(G, edges_src)
    src_data['edge_ids'] = edge_ids
    cost_matrix = ColumnDataSource(data={'G': G_matrix})

    # docs indicate that each value should be of the same length but this works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = ColumnDataSource(data=src_data)

    cost, clicked, error_msg = _get_create_divs(plot)

    code = PLOT_CREATE_JS
    on_click = code + "check_done()\nload_data()\nselect_edge()\ntree_update()"

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderer=edges_glyph, source=source, nodes_src=nodes_src,
               edges_src=edges_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    return _get_grid(plot, cost, error_msg, clicked)


def assisted_dijkstras_plot(G, s=0, **kw) -> GridBox:
    """Return a plot in which the user creates a shortest path tree from s.

    Args:
        G (nx.Graph): Networkx graph.
        s (int): The vertex to generate a shortest path tree from.

    Returns:
        GridBox: Plot in which the user creates a shortest path tree from s.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    nodes_src.data['fill_color'][s] = SECONDARY_COLOR
    nodes_src.data['line_color'][s] = SECONDARY_DARK_COLOR
    edges_src, edges_glyph = _add_edges(G, plot, show_labels=True)

    src_data = _get_blank_src_data(G)
    dist = [float('inf')]*len(G)
    dist[s] = 0
    src_data['dist'] = dist
    src_data['prev'] = [float('nan')]*len(G)
    src_data['settled'] = []
    src_data['frontier'] = [s]

    edge_ids, G_matrix = _edge_src_maps(G, edges_src)
    src_data['edge_ids'] = edge_ids
    cost_matrix = ColumnDataSource(data={'G': G_matrix})

    # docs indicate that each value should be of the same length but this works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = ColumnDataSource(data=src_data)

    error_msg = Div(text='', width=int(plot.width/3), align='center')

    table_data = {str(i): ['inf', '-'] for i in range(len(G))}
    table_data['index'] = ['label', 'prev']

    table_data[str(s)][1] = '0'
    table_src = ColumnDataSource(data=table_data)
    columns = [TableColumn(field='index', title='')]
    for i in range(len(G)):
        columns.append(TableColumn(field=str(i), title=str(i)))
    table = DataTable(source=table_src, columns=columns, height=80,
                      background='white', index_position=None, editable=False,
                      reorderable=False, sortable=False, selectable=False,
                      width=plot.width)

    on_click = PLOT_CREATE_JS + 'check_done()\nload_data()\ndijkstras()\n'

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderer=nodes_glyph, source=source, nodes_src=nodes_src,
               edges_src=edges_src, cost_matrix=cost_matrix,
               error_msg=error_msg, table_src=table_src)

    grid = gridplot([[plot],
                     [table],
                     [row(error_msg)]],
                    toolbar_location=None,
                    toolbar_options={'logo': None})

    return grid


def assisted_mst_algorithm_plot(G:nx.Graph, algorithm:str, **kw) -> GridBox:
    """Return a plot in which the user creates an MST with the given algorithm.

    Args:
        G (nx.Graph): Networkx graph.
        algorithm (str): {'prims', 'kruskals', 'reverse_kruskals'}

    Returns:
        GridBox: Plot in which the user creates an MST with given algorithm.
    """
    G = G.copy()
    plot = _blank_plot(G, **kw)

    _set_edge_positions(G)
    _set_graph_colors(G)

    nodes_src, nodes_glyph = _add_nodes(G, plot)
    if 's' in kw:
        nodes_src.data['fill_color'][kw['s']] = PRIMARY_DARK_COLOR
        nodes_src.data['line_color'][kw['s']] = PRIMARY_DARK_COLOR
    if algorithm == 'reverse_kruskals':
        hover_color = TERTIARY_COLOR
    else:
        hover_color = TERTIARY_DARK_COLOR
    edges_src, edges_glyph = _add_edges(G, plot, show_labels=True,
                                        hover_line_color=hover_color)

    src_data = _get_blank_src_data(G)
    if algorithm == 'prims':
        unvisited = list(range(len(G)))
        unvisited.remove(kw['s'])
        src_data['visited'] = [kw['s']]
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

    edge_ids, G_matrix = _edge_src_maps(G, edges_src)
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
    on_click = PLOT_CREATE_JS + code

    _add_tools(plot, on_click=on_click, nodes_glyph=nodes_glyph,
               renderer=edges_glyph, source=source, nodes_src=nodes_src,
               edges_src=edges_src, cost_matrix=cost_matrix, cost=cost,
               error_msg=error_msg, clicked=clicked)

    return _get_grid(plot, cost, error_msg, clicked)
