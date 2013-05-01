"""
This module contains several algorithms
that can be performed on graphs.
"""

from heap import PriorityQueue
from utils import make_graph_from_mst
import itertools


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
    attrs = graph.edge_attr
    entries = [(float(attrs[edge][0].weight[0]), edge[0], edge[1]) for edge in attrs]
    entries.sort(lambda a, b: cmp(a[0], b[0]))

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
            # union
            sets[u].update(sets[v])
            for ver in sets[u]:
                # set references to the specific union
                sets[ver] = sets[u]
            result += w
            length += 1

    print result


def prim(graph, start_node):
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
            temp_weight = float(graph.get_default_weights((cheapest_node, parent[cheapest_node]))[0])
            mst.append((temp_weight, (cheapest_node, parent[cheapest_node])))
            mst_sum += temp_weight
        for adj_node in graph.get_node_neighbours(cheapest_node):
            edge_weight = float(graph.get_default_weights((cheapest_node, adj_node))[0])
            if queue.contains_task(adj_node) and edge_weight < queue.get_priority(adj_node):
                parent[adj_node] = cheapest_node
                queue.add_task(task=adj_node, priority=edge_weight)

    print "Prim Weight: ", mst_sum
    return mst


def nearest_neighbor(graph, node):
    current_node = node
    visited_nodes = [node]
    tour_weight = 0
    weights = []

    while len(visited_nodes) < graph.get_node_count():
        neighbours = set(graph.get_node_neighbours(current_node))
        # all unvisited adjacent nodes
        for adj_node in neighbours.difference(visited_nodes):
            temp_weight = float(graph.get_default_weights((current_node, adj_node))[0])
            weights.append((temp_weight, (current_node, adj_node)))
        minedge = min(weights)
        tour_weight += minedge[0]
        current_node = minedge[1][1]
        visited_nodes.append(current_node)
        weights = []

    # add the last weight (weighted edge to the starting node)
    temp_weight = float(graph.get_default_weights((node, current_node))[0])

    print "Tour: ", visited_nodes
    print "Cost: ", tour_weight + temp_weight


def double_tree(graph):
    mst = prim(graph, graph.get_nodes()[0])
    index = 0
    tour_weight = 0

    mst_graph = make_graph_from_mst(mst, graph)
    res_tour = recursive_depth_first_search(mst_graph, mst_graph.get_nodes()[0])

    while index < len(res_tour) - 1:
        tour_weight += float(graph.get_default_weights((res_tour[index], res_tour[index + 1]))[0])
        index += 1

    tour_weight += float(graph.get_default_weights((res_tour[-1], res_tour[0]))[0])

    print "Tour: ", res_tour
    print "Cost: ", tour_weight

def branch_and_bound(graph):
    nodes = graph.get_nodes()
    upper_bound = 0
    index = 0

    #1. Get initial upper_bound
    while index < len(nodes) - 1:
        upper_bound += float(graph.get_default_weights((nodes[index], nodes[index + 1]))[0])
        index += 1

    upper_bound += float(graph.get_default_weights((nodes[-1], nodes[0]))[0])

    #2. Permutate & Branch
    for perm in itertools.permutations(nodes[1:]):
        temp_bound = 0
        index = 0
        broke = False
        perm = list(perm)
        perm.insert(0, nodes[0])

        while index < len(perm) - 1:
            temp_bound += float(graph.get_default_weights((perm[index], perm[index + 1]))[0])
            index += 1
            if temp_bound > upper_bound:
                broke = True
                #print temp_bound, "cut", upper_bound
                break

        if not broke:
            #print "not broke"
            temp_bound += float(graph.get_default_weights((perm[-1], perm[0]))[0])
            if temp_bound < upper_bound:
                upper_bound = temp_bound

    print upper_bound

    # 0 1 2
    # 0 2 1
    # 1 2 0
    # 1 0 2
    # 2 0 1
    # 2 1 0
    #1. Linear Depth

    

    #get edge_weight (current_node, node[0])
    #track_weight += edge_weight
    #if track_weight >= upper_bound -> next branch
    #if track_weight < upper_bound -> run into depth

    #1 Tiefensuche auf Ast
    #2 Backtracking
    #3 Obere Schranke setzen
def branch_bound_backtrack_start(graph):
    nodes = graph.get_nodes()
    print branch_bound_backtrack(graph, [nodes[0]], nodes[1:], 0, None)

def branch_bound_backtrack(graph, visited, unvisited, track_cost, minCost):
    visited.append(unvisited.pop(0))

    for perm in itertools.permutations(unvisited):
        print perm
        minCost = branch_bound_backtrack(graph, visited, list(perm), track_cost, minCost)

    if len(unvisited) <= 1:
        return minCost

    track_cost += float(graph.get_default_weights((visited[-2], visited[-1]))[0])
    if minCost is not None and track_cost >= minCost:
        return minCost

    if len(visited) == graph.get_node_count():
        track_cost += float(graph.get_default_weights((visited[-1], visited[0]))[0])
        if minCost is None or track_cost < minCost:
            minCost = track_cost
            return minCost

    #minCost = branch_bound_backtrack(graph, visited, unvisited, track_cost, minCost)




    return minCost


def permutate(nodelist):
    pool = tuple(nodelist)
    n = len(pool)
    for indices in product(range(n), repeat=n):
        if len(set(indices)) == n:
            yield tuple(pool[i] for i in indices)

    #0 1 2


