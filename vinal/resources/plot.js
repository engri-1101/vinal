// Used by plot_graph_iterations
var iteration = 0

// Used by plot_create
var G
var visited
var unvisited
var tree_edges
var edge_ids
var clicked_list
var i
var u
var v
var w

// Color Theme -- Using Google's Material Design Color System
// https://material.io/design/color/the-color-system.html

var PRIMARY_COLOR = '#1565c0'
var PRIMARY_LIGHT_COLOR = '#5e92f3'
var PRIMARY_DARK_COLOR = '#003c8f'
var SECONDARY_COLOR = '#d50000'
var SECONDARY_LIGHT_COLOR = '#ff5131'
var SECONDARY_DARK_COLOR = '#9b0000'
var PRIMARY_FONT_COLOR = '#ffffff'
var SECONDARY_FONT_COLOR = '#ffffff'
// Grayscale
var TERTIARY_COLOR = '#DFDFDF'
var TERTIARY_LIGHT_COLOR = 'white'  // Jupyter Notebook: white, Sphinx: #FCFCFC
var TERTIARY_DARK_COLOR = '#404040'


function increment_iteration() {
    if ((parseInt(n.text) + 1) < parseInt(k.text)) {
        n.text = (parseInt(n.text) + 1).toString()
    }
    iteration = parseInt(n.text)
}

function decrement_iteration() {
    if ((parseInt(n.text) - 1) >= 0) {
        n.text = (parseInt(n.text) - 1).toString()
    }
    iteration = parseInt(n.text)
}

function done_update() {
    if (iteration == parseInt(k.text) - 1) {
        done.text = "done."
    } else {
        done.text = ""
    }
}

function edge_subset_update() {
    edge_subset_src.data['xs'] = source.data['edge_xs'][iteration]
    edge_subset_src.data['ys'] = source.data['edge_ys'][iteration]
    edge_subset_src.change.emit()
}

function cost_update() {
    cost.text = source.data['costs'][iteration].toFixed(1)
}

function table_update() {
    table_src.data = source.data['tables'][iteration]
}

function nodes_update() {
    var in_tree = source.data['nodes'][iteration]

    for (let i = 0; i < nodes_src.data['line_color'].length ; i++) {
        if (in_tree.includes(i)) {
            nodes_src.data['fill_color'][i] = PRIMARY_DARK_COLOR
            nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
        } else {
            nodes_src.data['fill_color'][i] = PRIMARY_LIGHT_COLOR
            nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
        }
    }

    nodes_src.change.emit()
}

function swaps_update() {
    swaps_src.data['swaps_before_x'] = source.data['swaps_before_x'][iteration]
    swaps_src.data['swaps_before_y'] = source.data['swaps_before_y'][iteration]
    swaps_src.data['swaps_after_x'] = source.data['swaps_after_x'][iteration]
    swaps_src.data['swaps_after_y'] = source.data['swaps_after_y'][iteration]
    swaps_src.change.emit()
}

function check_done() {
    if (error_msg.text == 'done.') {
        return;
    }
}

function load_data() {
    G = cost_matrix.data['G']
    visited = source.data['visited']
    unvisited = source.data['unvisited']
    tree_edges = source.data['tree_edges']
    edge_ids = source.data['edge_ids']
    clicked_list = source.data['clicked']

    i = source.data['last_index']
    u = edges_src.data['u'][i]
    v = edges_src.data['v'][i]
    w = edges_src.data['weight'][i]
}

function select_edge() {
    if (!visited.includes(v)) {
        nodes_src.data['fill_color'][v] = PRIMARY_DARK_COLOR
        nodes_src.data['line_color'][v] = PRIMARY_DARK_COLOR
        visited.push(v)
        unvisited.splice(unvisited.indexOf(v), 1)
    }
    if (!visited.includes(u)) {
        nodes_src.data['fill_color'][u] = PRIMARY_DARK_COLOR
        nodes_src.data['line_color'][u] = PRIMARY_DARK_COLOR
        visited.push(u)
        unvisited.splice(unvisited.indexOf(u), 1)
    }
    edges_src.data['line_color'][i] = TERTIARY_DARK_COLOR
    tree_edges.push([u,v])
    clicked_list = tree_edges
    var prev_cost = parseFloat(cost.text)
    cost.text = (prev_cost + w).toFixed(1)
    error_msg.text = ''
}

function prims() {
    var possible = {}
    for (let i = 0; i < visited.length; i++) {
        for (let j = 0; j < unvisited.length; j++) {
            var a = visited[i]
            var b = unvisited[j]
            if (G[a][b] != 0) {
                var id = edge_ids[a][b]
                possible[id] = G[a][b]
            }
        }
    }

    var keys  = Object.keys(possible).sort(function(a,b) { return possible[a] - possible[b]; });
    var valid = keys.filter(function(x) { return possible[x] == possible[keys[0]]; });

    var first_edge = (visited.length == 0)
    var u_in_tree = visited.includes(u)
    var v_in_tree = visited.includes(v)
    var one_in_tree = (u_in_tree && !v_in_tree) || (!u_in_tree && v_in_tree)

    if (first_edge || one_in_tree) {
        if (valid.includes(i.toString())) {
            select_edge()
        } else {
            error_msg.text = 'Smaller weight edge exists.'
        }
    } else {
        if (u_in_tree && v_in_tree) {
            error_msg.text = 'Edge creates a cycle.'
        } else {
            error_msg.text = 'Edge not adjacent to the current tree.'
        }
    }
}

function kruskals() {
    var sorted_edges = source.data['sorted_edges']
    var forest = source.data['forest']
    var index = source.data['index'][0]

    var a = sorted_edges[index][0]
    var b = sorted_edges[index][1]
    while (forest[a] == forest[b]) {
        index += 1
        a = sorted_edges[index][0]
        b = sorted_edges[index][1]
    }
    var min_val = G[sorted_edges[index][0]][sorted_edges[index][1]]

    if (forest[u] != forest[v]) {
        if (G[u][v] == min_val) {
            select_edge()
            var x = forest[u]
            var y = forest[v]
            forest = forest.map(function(k) {if (k == y) {return x} else {return k} })
            source.data['forest'] = forest
            if (u == a && v == b) {
                index += 1
            }
        } else {
             error_msg.text = 'Smaller weight edge exists.'
        }
    } else {
        error_msg.text = 'This edge creates a cycle.'
    }

    source.data['index'][0] = index
}

function reverse_kruskals() {
    var sorted_edges = source.data['sorted_edges']
    var index = source.data['index'][0]

    function is_connected(G) {
        function fillArray(value, len) {
            var a = [value];
            while (a.length * 2 <= len) a = a.concat(a);
            if (a.length < len) a = a.concat(a.slice(0, len - a.length));
            return a;
        }

        var check = fillArray(false, G.length)
        check[0] = true
        var checked = [0]

        while (checked.length > 0) {
            var a = checked.shift()
            for (let b = 0; b < G[a].length; b++) {
                if (G[a][b] > 0 && !check[b]) {
                    check[b] = true
                    checked.push(b)
                }
            }
        }

        return check.every(function(x) {return x})
    }

    var a = sorted_edges[index][0]
    var b = sorted_edges[index][1]
    var tmp_w = G[a][b]
    G[a][b] = 0
    G[b][a] = 0
    while (!is_connected(G) || !tree_edges.includes(edge_ids[a][b])) {
        G[a][b] = tmp_w
        G[b][a] = tmp_w
        index += 1
        a = sorted_edges[index][0]
        b = sorted_edges[index][1]
        tmp_w = G[a][b]
        G[a][b] = 0
        G[b][a] = 0
    }
    G[a][b] = tmp_w
    G[b][a] = tmp_w
    var max_val = tmp_w

    var tmp_w = w
    G[u][v] = 0
    G[v][u] = 0

    let e = tree_edges.indexOf(i)
    if (e > -1) {
        if (is_connected(G)) {
            if (tmp_w == max_val) {
                tree_edges.splice(e, 1);
                if (u == a && v == b) {
                    index += 1
                }
                edges_src.data['line_color'][i] = TERTIARY_COLOR
                var prev_cost = parseFloat(cost.text)
                cost.text = (prev_cost - w).toFixed(1)
                error_msg.text = ''
            } else {
                G[u][v] = tmp_w
                G[v][u] = tmp_w
                error_msg.text = 'Larger edge weight exists. Try ('
                                  .concat(a.toString())
                                  .concat(', ')
                                  .concat(b.toString())
                                  .concat(').')
            }
        } else {
            G[u][v] = tmp_w
            G[v][u] = tmp_w
            error_msg.text = 'Removing this edge disconnects the graph.'
        }
    } else {
        G[u][v] = tmp_w
        G[v][u] = tmp_w
        error_msg.text = 'Edge already removed.'
    }

    var u_list = edges_src.data['u']
    var v_list = edges_src.data['v']
    clicked_list = []
    for (let i = 0; i < tree_edges.length; i++) {
        var edge = tree_edges[i]
        clicked_list.push([u_list[edge], v_list[edge]])
    }
    source.data['index'][0] = index
}

function tree_update() {
    if (tree_edges.length == nodes_src.data['x'].length - 1) {
        error_msg.text = 'done.'
    }

    clicked.text = '['
    for (let i = 0; i < clicked_list.length; i++) {
        var edge_str = clicked_list[i].join(',')
        clicked.text = clicked.text.concat('(').concat(edge_str).concat(')')
        if (!(i == clicked_list.length - 1)) {
            clicked.text = clicked.text.concat(',')
        }
    }
    clicked.text = clicked.text.concat(']')

    source.change.emit()
    nodes_src.change.emit()
    edges_src.change.emit()
}

function create_tour_on_click () {
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
        nodes_src.data['line_color'][v] = PRIMARY_DARK_COLOR
        nodes_src.data['fill_color'][v] = PRIMARY_DARK_COLOR

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
}

// BEGIN FUNCTION CALLING
