/** @type {float[][]} */
var G
/** @type {integer[]} */
var visited
/** @type {integer[]} */
var unvisited
/** @type {integer[][]} */
var tree_edges
/** @type {integer[][]} */
var edge_ids
/** @type {(integer[]|integer[][])} */
var clicked_list
/** @type {integer} */
var i
/** @type {integer} */
var u
/** @type {integer} */
var v
/** @type {float} */
var w


/**
 * Check if the error message text indicates tour or tree is complete.
 */
function check_done() {
    if (error_msg.text == 'done.') {
        return;
    }
}

/**
 * Load the data necessary data for MST and shortest path tree creation.
 */
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

/**
 * Update node and edge highlighting upon a proper edge selection.
 */
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

/**
 * Iteration of Prim's algorithm.
 * Check if user-selected edge is a valid next edge. If valid, add the selected
 * edge. If not, an error message is returned.
 */
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

/**
 * Iteration of Kruskal's algorithm.
 * Check if user-selected edge is a valid next edge. If valid, add the selected
 * edge. If not, an error message is returned.
 */
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

/**
 * Return an array of length len where every element has the given value.
 * @param {float} value
 * @param {integer} len
 * @returns {float[]}
 */
function fillArray(value, len) {
    var a = [value];
    while (a.length * 2 <= len) a = a.concat(a);
    if (a.length < len) a = a.concat(a.slice(0, len - a.length));
    return a;
}

/**
 * Return true iff the undirected graph G is connected.
 * @param {float[][]} G
 * @returns {boolean}
 */
function is_connected(G) {
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

/**
 * Iteration of reverse Kruskal's algorithm.
 * Check if user-selected edge is a valid next edge. If valid, add the selected
 * edge. If not, an error message is returned.
 */
function reverse_kruskals() {
    var sorted_edges = source.data['sorted_edges']
    var index = source.data['index'][0]

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

/**
 * Final updates when creating an MST or shortest path tree.
 */
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

/**
 * Check if user-selected node is valid and add to tour if valid.
 */
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

/**
 * Iteration of Dijkstra's algorithm.
 * Check if user-selected node is a valid next node. If valid, add the selected
 * node. If not, an error message is returned.
 */
function dijkstras() {
    var dist = source.data['dist']
    var prev = source.data['prev']
    var S = source.data['settled']
    var F = source.data['frontier']

    F.sort(function(a,b) {return dist[a] - dist[b]})

    if (F.includes(i)) {
        if (dist[i] == dist[F[0]]) {
            var f = i
            F.splice(F.indexOf(f), 1)
            S.push(f)
            for (let w = 0; w < G.length; w++) {
                if (G[f][w] != 0) {
                    if (!S.includes(w) && !F.includes(w)) {
                        dist[w] = dist[f] + G[f][w]
                        prev[w] = f
                        F.push(w)
                        var k = edge_ids[w][f]
                        edges_src.data['line_color'][k] = TERTIARY_DARK_COLOR
                    } else {
                        if (dist[f] + G[f][w] < dist[w]) {
                            var k = edge_ids[prev[w]][w]
                            edges_src.data['line_color'][k] = TERTIARY_COLOR

                            dist[w] = dist[f] + G[f][w]
                            prev[w] = f

                            var k = edge_ids[f][w]
                            edges_src.data['line_color'][k] = TERTIARY_DARK_COLOR
                        }
                    }
                }
            }
            error_msg.text = ''
        } else {
            error_msg.text = 'A closer frontier node exists.'
        }
    } else {
        error_msg.text = 'This node is not in the frontier set.'
    }

    for (let i = 0; i < G.length; i++) {
        // Highlight settled and frontier nodes
        if (S.includes(i)) {
            nodes_src.data['fill_color'][i] = PRIMARY_DARK_COLOR
            nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
        } else if (F.includes(i)) {
            nodes_src.data['fill_color'][i] = SECONDARY_COLOR
            nodes_src.data['line_color'][i] = SECONDARY_DARK_COLOR
        } else {
            nodes_src.data['fill_color'][i] = PRIMARY_LIGHT_COLOR
            nodes_src.data['line_color'][i] = PRIMARY_DARK_COLOR
        }

        // Update table source
        var dist_text
        var prev_text
        if (isFinite(dist[i])) {
            dist_text = dist[i].toFixed(1)
        } else {
            dist_text = 'inf'
        }
        if (S.includes(i)) {
            dist_text = dist_text.concat('*')
        }

        if (isNaN(prev[i])) {
            prev_text = '-'
        } else {
            prev_text = prev[i]
        }
        table_src.data[i.toString()] = [dist_text, prev_text]
    }

    if (F.length == 0) {
        error_msg.text = 'done.'
    }

    source.change.emit()
    nodes_src.change.emit()
    edges_src.change.emit()
    table_src.change.emit()
}

// BEGIN FUNCTION CALLING
