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
    sets = [[n] for n in graph.get_nodes()]
    mst = []
    mst_sum = 0

    for cheapest in entries:
        w, u, v = cheapest[0], cheapest[1], cheapest[2]
        res1, res2 = find_in_set(sets, u), find_in_set(sets, v)
        if res1 != res2:
            sets = union(sets, res1, res2)
            mst.append((w, u, v))
            mst_sum += w
            if len(mst) == graph.get_node_count() - 1:
                break

    print mst_sum


def find_in_set(sets, node):
    idx = 0
    for curr_set in sets:
        if node in curr_set:
            return (curr_set, idx)
        idx += 1


def union(sets, res1, res2):
    new_set = [res1[0] + res2[0]] + [b for a, b in enumerate(sets) if a not in [res1[1], res2[1]]]
    return new_set


def cmp_func(a, b):
    return cmp(a[0], b[0])
