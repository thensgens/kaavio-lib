"""
This module contains several algorithms
that can be performed on (directed) graphs.
"""


def recursive_depth_first_search(graph, node, visited_nodes=[]):
    visited_nodes.append(node)
    for each in graph.get_node_neighbours(node):
        if each not in visited_nodes:
            recursive_depth_first_search(graph, each, visited_nodes)

    return visited_nodes


def iterative_breadth_first_search(graph, node):
    queue = []
    visited_nodes = []

    # push the start node into the queue
    queue.append(node)

    while queue:
        curr_element = queue.pop(0)
        if curr_element not in visited_nodes:
            visited_nodes.append(curr_element)

        for neighbour in graph.get_node_neighbours(curr_element):
            if neighbour not in visited_nodes:
                queue.append(neighbour)

    return visited_nodes


def get_coherent_components_count(graph):
    bfs_results = []
    for node in graph.get_nodes():
        bfs_res, is_coherent = is_graph_coherent(graph, node)
        if is_coherent:
            return 1

        for el in bfs_results:
            if set(el) != set(bfs_res):
                bfs_results.append(bfs_res)

    return len(bfs_results)


def is_graph_coherent(graph, node):
    iter_bfs_res = iterative_breadth_first_search(graph, node)
    return iter_bfs_res, len(iter_bfs_res) == graph.get_node_count()
