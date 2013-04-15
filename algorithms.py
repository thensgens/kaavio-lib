"""
This module contains several algorithms
that can be performed on graphs.
"""

from heap import PriorityQueue

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

def kruskal_2(graph):
    attrs  = graph.edge_attr
    entries = [(float(attrs[edge][0].weight[0]), edge[0], edge[1]) for edge in attrs]
    entries.sort(cmp_func)

    outedges = []
    result = 0
    edgecount = graph.get_node_count() - 1
    length = 0

    sets = dict((n, set([n])) for n in graph.get_nodes())

    while length < edgecount:
        edge = entries.pop(0)
        w, u, v = edge

        if sets[u] != sets[v]:
            outedges.append(edge)
            sets[u].update(sets[v])
            for ver in sets[u]:
                sets[ver] = sets[u]
            result += w
            length += 1

    print result


def prim_wiki(graph, start_node):
    queue = PriorityQueue()
    parent = {}
    mst = []
    mst_sum = 0

    for node in graph.get_nodes():
        queue.add_task(task=node, priority=PriorityQueue.INFINITY)
        parent[node] = None

    # put first node in the queue
    queue.add_task(task=start_node, priority=0)

    while queue.not_empty():
        cheapest_node = queue.pop_task()
        if parent[cheapest_node] is not None:
            mst.append((cheapest_node, parent[cheapest_node]))
            mst_sum += float(graph.get_default_weights((cheapest_node, parent[cheapest_node]))[0])
        for adj_node in graph.get_node_neighbours(cheapest_node):
            edge_weight = float(graph.get_default_weights((cheapest_node, adj_node))[0])
            if queue.contains_task(adj_node) and edge_weight < queue.get_priority(adj_node):
                parent[adj_node] = cheapest_node
                queue.add_task(task=adj_node, priority=edge_weight)

    print mst_sum


def prim(graph, start_node):
    queue = PriorityQueue()
    update_queue(graph, start_node, queue)

    while queue.not_empty():
        cheapest_edge = queue.pop_task()
        update_queue(graph, cheapest_edge, queue)


def update_queue(graph, node, queue):
    adj_nodes = graph.get_node_neighbours(node)
    for adj_node in adj_nodes:
        prio = graph.edge_attr[(node, adj_node)][0].weight[0]
        queue.add_task(task=(node, adj_node), priority=prio)


def cmp_func(a, b):
    return cmp(a[0], b[0])
