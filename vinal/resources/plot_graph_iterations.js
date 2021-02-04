/** @type {integer} */
var iteration = 0


/**
 * Increment iteration and update n Div.
 */
function increment_iteration() {
    if ((parseInt(n.text) + 1) < parseInt(k.text)) {
        n.text = (parseInt(n.text) + 1).toString()
    }
    iteration = parseInt(n.text)
}

/**
 * Decrement iteration and update n Div.
 */
function decrement_iteration() {
    if ((parseInt(n.text) - 1) >= 0) {
        n.text = (parseInt(n.text) - 1).toString()
    }
    iteration = parseInt(n.text)
}

/**
 * Update which nodes are highlighted given the current iteration.
 */
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

/**
 * Update which subset of edges are highlighted given the current iteration.
 */
function edge_subset_update() {
    edge_subset_src.data['xs'] = source.data['edge_xs'][iteration]
    edge_subset_src.data['ys'] = source.data['edge_ys'][iteration]
    edge_subset_src.change.emit()
}

/**
 * Update which swaps are updated given the current iteration
 */
function swaps_update() {
    swaps_src.data['swaps_before_x'] = source.data['swaps_before_x'][iteration]
    swaps_src.data['swaps_before_y'] = source.data['swaps_before_y'][iteration]
    swaps_src.data['swaps_after_x'] = source.data['swaps_after_x'][iteration]
    swaps_src.data['swaps_after_y'] = source.data['swaps_after_y'][iteration]
    swaps_src.change.emit()
}

/**
 * Update which table is shown given the current iteration.
 */
function table_update() {
    table_src.data = source.data['tables'][iteration]
}

/**
 * Update which cost is shown given the current iteration.
 */
function cost_update() {
    cost.text = source.data['costs'][iteration].toFixed(1)
}

/**
 * Update done Div based on the current iteration.
 */
function done_update() {
    if (iteration == parseInt(k.text) - 1) {
        done.text = "done."
    } else {
        done.text = ""
    }
}

// BEGIN FUNCTION CALLING
