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
