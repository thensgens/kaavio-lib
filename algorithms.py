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
    # workaround to enter the for-loop below
    bfs_results.append([])
    #for node in graph.get_nodes():
    nodelist = set(graph.get_nodes())
    while nodelist:
        #print "Node %s: " % node
        bfs_res, is_coherent = is_graph_coherent(graph, nodelist.pop())
        if is_coherent:
            #print "BFS result %s valid; breaks loop" % bfs_res
            return 1

        if all(set(item) != set(bfs_res) for item in bfs_results):
            bfs_results.append(bfs_res)
            nodelist -= set(bfs_res)
            #print "New Element: " + str(len(bfs_results)-1)
        print "Aktuelles result %d\tNode Count %d " % (len(bfs_res), graph.get_node_count())

    return len(bfs_results) - 1


def is_graph_coherent(graph, node):
    #iter_bfs_res = iterative_breadth_first_search(graph, node)
    #return iter_bfs_res, len(iter_bfs_res) == graph.get_node_count()
    recur_bfs_res = recursive_depth_first_search(graph, node, [])
    return recur_bfs_res, len(recur_bfs_res) == graph.get_node_count()
