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

    # 1. Get initial upper_bound
    while index < len(nodes) - 1:
        upper_bound += float(graph.get_default_weights((nodes[index], nodes[index + 1]))[0])
        index += 1

    upper_bound += float(graph.get_default_weights((nodes[-1], nodes[0]))[0])

    # 2. Permutate & Branch
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


"""  Branch-and-Bound """

#global variables for testing purposes
best_solution = 99999.
current_cost = 0.0
visited_nodes = []
visited_dict = {}

def branch_bound_backtrack_start(graph):
    nodes = graph.get_nodes()
    result = []
    path = []

    # start node for BnB
    start = nodes[0]

    # initially all nodes are unvisited
    for node in nodes:
        visited_dict[node] = False

    # start BnB-Algorithm w/ Backtracking
    branch_bound_backtrack(graph, start, start, start, result, path)
    print best_solution


def branch_bound_backtrack(graph, last, current, start, result, path):
    global best_solution
    global current_cost
    global visited_nodes
    global visited_dict

    visited_dict[current] = True
    if current not in visited_nodes:
        visited_nodes.append(current)

    try:
        temp_cost = current_cost + float(graph.get_default_weights((last, current))[0])
        if temp_cost > best_solution:
            visited_dict[current] = False
            visited_nodes.remove(current)
            return
        current_cost = temp_cost
        path.append((last, current))
    except KeyError:
        pass

    all_visited = len(visited_nodes) == graph.get_node_count()

    if all_visited and visited_dict[start] and current == start:
        del result[:]
        result.extend(path)
        best_solution = current_cost
        current_cost -= float(graph.get_default_weights((last, current))[0])
        path.pop(len(path) - 1)
        visited_dict[current] = False
        return

    for next in graph.get_node_neighbours(current):
        if not visited_dict[next] or (all_visited and next == start):
            branch_bound_backtrack(graph, current, next, start, result, path)

    try:
        current_cost -= float(graph.get_default_weights((last, current))[0])
        path.pop(len(path) - 1)
    except KeyError:
        pass

    visited_dict[current] = False
    try:
        visited_nodes.remove(current)
    except:
        pass





















# if not unvisited:
#     return
# visited.append(unvisited.pop(0))

# # end of recursion
# if len(visited) == graph.get_node_count():
#     track_cost += float(graph.get_default_weights((visited[-1], visited[0]))[0])
#     if track_cost < min(costs):
#         costs.append(track_cost)
#         return

# for perm in itertools.permutations(unvisited):
#     branch_bound_backtrack(graph, visited[:], list(perm), track_cost, costs)

# track_cost += float(graph.get_default_weights((visited[-2], visited[-1]))[0])
# if costs is not None and track_cost >= costs:
#     return costs


# return costs


# def permutate(nodelist):
#     pool = tuple(nodelist)
#     n = len(pool)
#     for indices in product(range(n), repeat=n):
#         if len(set(indices)) == n:
#             yield tuple(pool[i] for i in indices)