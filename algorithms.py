"""
This module contains several algorithms
that can be performed on graphs.
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

    # push the first node into the queue
    queue.append(node)
    visited_nodes.append(node)

    while queue:
        curr_element = queue.pop(0)
        for neighbour in graph.get_node_neighbours(curr_element):
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour)
                queue.append(neighbour)

    return visited_nodes


def get_coherent_components_count(graph):
    trav_results = []
    # insert dummy item to enter the for-loop below
    trav_results.append([])
    nodelist = set(graph.get_nodes())
    while nodelist:
        trav_result, is_coherent = is_graph_coherent(graph, nodelist.pop())
        if is_coherent:
            return 1

        if all(set(item) != set(trav_result) for item in trav_results):
            trav_results.append(trav_result)
            nodelist -= set(trav_result)

    return len(trav_results) - 1


def is_graph_coherent(graph, node):
    trav_result = recursive_depth_first_search(graph, node, [])
    #trav_result = iterative_breadth_first_search(graph, node)
    return trav_result, len(trav_result) == graph.get_node_count()


def kruskal(graph):
    attrs  = graph.edge_attr
    entries = [(float(attrs[edge][0].weight[0]), edge[0], edge[1]) for edge in attrs]
    entries.sort(cmp_func)

    A = []
    mengen = []
    for node in graph.get_nodes():
        mengen.append(set([node]))

    for edge in entries:
        if mengen[mengen.index(set([edge[1]]))] != mengen[mengen.index(set([edge[2]]))]:
           A.append(edge)
           mengen[mengen.index(set([edge[1]]))] |= mengen[mengen.index(set([edge[2]]))]
           #mengen.remove(set([edge[2]]))

    print A
def cmp_func(a, b):
    return cmp(a[0], b[0])
