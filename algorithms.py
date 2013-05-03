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
                # set references to the specific union (slooow!)
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


"""  Start corresponding Branch-and-Bound or Brute-Force algorithms."""
def start_bnb_bruteforce(graph, bnb=True):
    nodes = graph.get_nodes()
    curr_path = []

    """
        Branch-and-Bound Property (holds necessary values for the computation)
    """
    class BnB_Property(object):
        def __init__(self, best, result_path, current_cost, visited):
            self.__best = best
            self.__result_path = result_path
            self.__current_cost = current_cost
            self.__nodes = nodes
            self.__visited = visited

        def get_best(self): return self.__best
        def set_best(self, best): self.__best = best
        def get_result_path(self): return self.__result_path
        def set_result_path(self, result_path): self.__result_path = result_path
        def get_current_cost(self): return self.__current_cost
        def set_current_cost(self, current_cost): self.__current_cost = current_cost
        def get_visited(self): return self.__visited
        def set_visited(self, visited): self.__visited = visited

        best = property(get_best, set_best)
        result_path = property(get_result_path, set_result_path)
        current_cost = property(get_current_cost, set_current_cost)
        visited = property(get_visited, set_visited)

    # create BnB property object (w/ default initialize values)
    bnb_prop = BnB_Property(99999., [], 0., {})

    # set the start node for BnB
    start = nodes[0]

    # initially all nodes are unvisited
    for node in nodes:
        bnb_prop.visited[node] = False

    # start BnB-Algorithm w/ backtracking (default); if not start brute force algorithm
    if bnb:
        branch_bound_backtrack(graph, start, start, start, curr_path, bnb_prop)
    else:
        brute_force(graph, start, start, start, curr_path, bnb_prop)

    # print the result path and the actual result (sum of edge weights)
    print bnb_prop.result_path
    print bnb_prop.best


def branch_bound_backtrack(graph, last, current, start, curr_path, bnb_prop):
    bnb_prop.visited[current] = True

    try:
        # sum the edge's (last, current) cost to the current path costs
        temp_cost = bnb_prop.current_cost + float(graph.get_default_weights((last, current))[0])

        # if the path cost is already higher than the (temporarily) best value (upper bound) -> STOP.
        if temp_cost > bnb_prop.best:
            bnb_prop.visited[current] = False
            return

        bnb_prop.current_cost = temp_cost
        curr_path.append((last, current))
    except KeyError:
        # this exception handles the disallowed access of non-present edges (e.g. (0, 0))
        pass

    all_nodes_visited = all(bnb_prop.visited.values())

    if current == start and all_nodes_visited:
        # now we should reset the old solution (because we have a better one) and set its value
        # to the new one; additionally remove the current element -> permutation
        del bnb_prop.result_path[:]
        bnb_prop.result_path.extend(curr_path)
        bnb_prop.best = bnb_prop.current_cost
        bnb_prop.current_cost -= float(graph.get_default_weights((last, current))[0])
        curr_path.pop(len(curr_path) - 1)
        bnb_prop.visited[current] = False
        return

    for next in graph.get_node_neighbours(current):
        last_step = all_nodes_visited and next == start
        if not bnb_prop.visited[next] or last_step:
            branch_bound_backtrack(graph, current, next, start, curr_path, bnb_prop)

    try:
        bnb_prop.current_cost -= float(graph.get_default_weights((last, current))[0])
        curr_path.pop(len(curr_path) - 1)
    except KeyError:
        pass

    bnb_prop.visited[current] = False


def brute_force(graph, last, current, start, curr_path, bnb_prop):
    bnb_prop.visited[current] = True

    try:
        # sum the edge's (last, current) cost to the current path costs
        temp_cost = bnb_prop.current_cost + float(graph.get_default_weights((last, current))[0])
        bnb_prop.current_cost = temp_cost
        curr_path.append((last, current))
    except KeyError:
        pass

    all_nodes_visited = all(bnb_prop.visited.values())

    if all_nodes_visited and current == start:
        if bnb_prop.current_cost <= bnb_prop.best:
            del bnb_prop.result_path[:]
            bnb_prop.result_path.extend(curr_path)
            bnb_prop.best = bnb_prop.current_cost

        bnb_prop.current_cost -= float(graph.get_default_weights((last, current))[0])
        curr_path.pop(len(curr_path) - 1)
        bnb_prop.visited[current] = False
        return

    for next in graph.get_node_neighbours(current):
        last_step = all_nodes_visited and next == start
        if not bnb_prop.visited[next] or last_step:
            branch_bound_backtrack(graph, current, next, start, curr_path, bnb_prop)

    try:
        bnb_prop.current_cost -= float(graph.get_default_weights((last, current))[0])
        curr_path.pop(len(curr_path) - 1)
    except KeyError:
        pass

    bnb_prop.visited[current] = False


def brute_force_itertools(graph):
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
            # if temp_bound > upper_bound:
            #     broke = True
            #     #print temp_bound, "cut", upper_bound
            #     break

        if not broke:
            temp_bound += float(graph.get_default_weights((perm[-1], perm[0]))[0])
            if temp_bound < upper_bound:
                upper_bound = temp_bound

    print upper_bound
