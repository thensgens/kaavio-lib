"""
This module contains several algorithms
that can be performed on graphs.
"""

from heap import PriorityQueue
from graph_utils import make_graph_from_mst
from graph import Graph
from basegraph import EdgeProperty, NodeProperty
import itertools


def recursive_depth_first_search(graph, node, visited_nodes=[], target=None):
    visited_nodes.append(node)
    for each in graph.get_node_neighbours(node):
        if each not in visited_nodes:
            if target not in visited_nodes:
                recursive_depth_first_search(graph, each, visited_nodes)

    return visited_nodes


def iterative_breadth_first_search(graph, node, target=None):
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
                if neighbour == target:
                    return visited_nodes
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
        queue.add_task(task=node, priority=float('Inf'))
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


def start_bnb_bruteforce(graph, bnb=True):
    """
        Start corresponding Branch-and-Bound or Brute-Force algorithms.
    """
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


def dijkstra(graph, start, end=None):
    nodes = graph.get_nodes()
    # initialize distance dictionary

    dist = {node: float('Inf') for node in nodes}
    dist[start] = 0
    # initialize predecessor dictionary
    pred = {node: None for node in nodes}

    # initialize prio queue for "unvisited nodes
    nodes_nonfinal = PriorityQueue()
    for node in nodes:
        nodes_nonfinal.add_task(task=node, priority=dist[node])

    # main computation loop
    while nodes_nonfinal.not_empty():
        u = nodes_nonfinal.pop_task()
        for adj in graph.get_node_neighbours(u):
            if nodes_nonfinal.contains_task(adj):
                temp = dist[u] + float(graph.get_default_weights((u, adj))[0])
                if temp < dist[adj]:

                    pred[adj] = u
                    nodes_nonfinal.add_task(task=adj, priority=temp)

    # if an end node was specified, the corresponding shortest path shall be
    # computed and displayed
    if end is not None:
        path, path_sum = shortest_path(graph, pred, end)
        #print '#' * 50
        #print 'Path: ', path
        #print 'Weight: ', path_sum
        return path
    else:
        get_shortest_path_tree(graph, pred, start)


def bellman_ford(graph, start, end=None, neg_cycle_detect=False):
    # initialize necessary data structures
    dist = {}
    pred = {}
    for node in graph.get_nodes():
        dist[node] = float('Inf')
        pred[node] = None
    dist[start] = 0
    pred[start] = start

    #optimized_nodelist = iterative_breadth_first_search(graph, start)

    # optimized main computation loop & cycle detection
    updated = False
    res_cycle_node = None

    for idx in range(graph.get_node_count()):
        updated = False
        potential_cycle_node = None
        #for u in optimized_nodelist:
        for u in graph.get_nodes():
            for v in graph.get_node_neighbours(u):
                temp = float(graph.get_default_weights((u, v))[0])
                if dist[u] + temp < dist[v]:
                    dist[v] = dist[u] + temp
                    pred[v] = u
                    updated = True
                    potential_cycle_node = v

        if not updated:
            break

        if idx + 1 == graph.get_node_count() and updated:
            res_cycle_node = potential_cycle_node
            break

    if end is not None:
        path, path_sum = shortest_path(graph, pred, end)
        return path
    elif not neg_cycle_detect:
        get_shortest_path_tree(graph, pred, start)
    else:
        return get_cycle_nodes(graph, pred, res_cycle_node)


def get_cycle_nodes(graph, pred, res_cycle_node):
    cycle_nodes = []
    if res_cycle_node is not None:
        # inner cycle
        for idx in xrange(graph.get_node_count()):
            res_cycle_node = pred[res_cycle_node]
        current_cycle_node = pred[res_cycle_node]
        while current_cycle_node != res_cycle_node:
            cycle_nodes.insert(0, current_cycle_node)
            current_cycle_node = pred[current_cycle_node]
        cycle_nodes.append(res_cycle_node)
    return cycle_nodes


def shortest_path(graph, pred, end):
    path = [end]
    visited = [end]
    u = end
    path_sum = 0
    while pred[u] is not None and pred[u] not in visited:
        path_sum += float(graph.get_default_weights((pred[u], u))[0])
        u = pred[u]
        visited.append(u)
        path.insert(0, u)

    return path, path_sum


def get_shortest_path_tree(graph, pred, start):
    start = int(start)
    visited = [start]
    next = [(None, start)]

    print ""
    print '-' * 40
    print "Startnode: ", start
    print '-' * 40

    next2 = next
    while True:
        next = next2
        next2 = []
        for e in pred:
            for f in next:
                if pred[e] == f[1]:
                    next2.append((f[1], e))
                    visited.append(e)
        if len(next2) <= 0:
            break
        for ele in next2:
            x, path_sum = shortest_path(graph, pred, ele[1])
            print ele[0], "->", ele[1], "Cost from Startnode: ", path_sum
        print '-' * 40

    unvisited = set(pred.keys()) - set(visited)
    if len(unvisited) > 0:
        print "Unvisited", list(unvisited)
        print '-' * 40


def make_residual_graph(graph, cap_index=0, flow_index=1):
    resGraph = Graph(directed=True)
    for node in graph.get_nodes():
        resGraph.add_nodes((node, None))

    for edge in graph.get_edges():
        maxCapa = float(graph.get_default_weights(edge)[cap_index])
        currentCapa = float(graph.get_default_weights(edge)[flow_index])
        backEdge = (edge[1], edge[0])

        if currentCapa > 0:
            atr = EdgeProperty(wgt=[currentCapa])
            resGraph.add_edges([backEdge, atr])
        if currentCapa < maxCapa:
            atr = EdgeProperty(wgt=[maxCapa-currentCapa])
            resGraph.add_edges([edge, atr])

    return resGraph


def update_graph_from_path_ssp(graph, path, gamma):
    result = graph
    for e in path:
        back_e = (e[1], e[0])

        if e in result.get_edges():
            result.get_default_weights(e)[2] += gamma
            result.get_node_weights(e[0])[1] += gamma
            result.get_node_weights(e[1])[1] -= gamma
        elif back_e in result.get_edges():
            result.get_default_weights(back_e)[2] -= gamma
            result.get_node_weights(back_e[0])[1] -= gamma
            result.get_node_weights(back_e[1])[1] += gamma

    return result


def make_graph_from_residual(graph, path, gamma, edge_weight_index=2):

    graph_edges = graph.get_edges()

    for e in path:
        back_e = (e[1], e[0])

        if e in graph_edges:
            edge_weight = graph.get_default_weights(e)
            edge_weight[edge_weight_index] += gamma

        elif back_e in graph_edges:
            edge_weight = graph.get_default_weights(back_e)
            edge_weight[edge_weight_index] -= gamma

    return graph


def edmonds_karp(graph, source, target, cap_index=0, flow_index=1):
    work_graph = graph

    while True:
        index = 0
        edges = []
        path = []
        resid = make_residual_graph(work_graph, cap_index, flow_index)

        path = bfs(resid, source, target)
        if path is None:
            break

        while index < len(path) - 1:
            edges.append((path[index], path[index + 1]))
            index += 1
        gamma = min(edges, key=lambda edge: float(resid.get_default_weights(edge)[0]))
        gamma = float(resid.get_default_weights(gamma)[0])
        work_graph = make_graph_from_residual(graph, edges, gamma, flow_index)

    flow = 0
    for node in work_graph.get_node_neighbours(source):
        flow += float(graph.get_default_weights((source, node))[1])
    return work_graph


def backtrace(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs(graph, start, end):
    parent = {}
    visited = {node: False for node in graph.get_nodes()}
    queue = []
    queue.append(start)
    while queue:
        node = queue.pop(0)
        if node == end:
            return backtrace(parent, start, end)
        for adjacent in graph.get_node_neighbours(node):
            if not visited[adjacent]:
                visited[adjacent] = True
                parent[adjacent] = node
                queue.append(adjacent)


def cycle_cancelling(graph):
    """
        get_default_weights:
            index = 0   =>      cost
            index = 1   =>      capacity
            index = 2   =>      flow
    """
    # the return value of this function (represents the minimal cost flow)
    result_flow_cost = 0.

    # create s* (super source) and t* (super target)
    ss = -1
    ts = -2

    s_list, t_list = [], []
    # if node A's balance is > 0 => edge (s*, A)
    # if node A's balance is < 0 => edge (A, t*)
    for node in graph.get_nodes():
        if graph.get_node_weights(node)[0] > 0:
            s_list.append(node)
        elif graph.get_node_weights(node)[0] < 0:
            t_list.append(node)

    # STEP 1: create b-flow
    # add s* and t* to the graph (balance is infinity)
    graph.add_nodes([ss, NodeProperty(wgt=[float('Inf')])])
    graph.add_nodes([ts, NodeProperty(wgt=[float('Inf')])])

    # add all edges due to the creation of s* and t*
    for s in s_list:
        edge_prop = EdgeProperty(wgt=[0., graph.get_node_weights(s)[0], 0.])
        graph.add_edges([(ss, s), edge_prop])
    for t in t_list:
        edge_prop = EdgeProperty(wgt=[0., graph.get_node_weights(t)[0] * -1, 0.])
        graph.add_edges([(t, ts), edge_prop])

    # compute the maximal flow from s* to t*
    graph = edmonds_karp(graph, ss, ts, cap_index=1, flow_index=2)

    b_flow = True
    for s in s_list:
        edge_obj = graph.get_default_weights((ss, s))
        if edge_obj[2] != edge_obj[1]:
            b_flow = False
        graph.remove_edge((ss, s))

    for t in t_list:
        edge_obj = graph.get_default_weights((t, ts))
        if edge_obj[2] != edge_obj[1]:
            b_flow = False
        graph.remove_edge((t, ts))

    if not b_flow:
        print "Error. No b-Flow!"
        return

    # remove s* and t* from the graph
    graph.remove_node(ss)
    graph.remove_node(ts)

    # STEP 2-4
    while True:
        resid, back_edges = make_residual_graph_ssp(graph)

        # step 3 here
        visited_nodes = []
        for node in set(graph.get_nodes()) - set(visited_nodes):
            cycle_nodes = bellman_ford(resid, start=node, neg_cycle_detect=True)
            for n in cycle_nodes:
                visited_nodes.append(n)
            if len(cycle_nodes) > 2:
                break

        # if no negative cycle could be found -> STOP.
        if len(cycle_nodes) <= 2:
            result_flow_cost = get_flow_cost(graph)
            break

        cycle_nodes.append(cycle_nodes[0])

        # step 4 here
        min_cap = get_min_resid_cap(resid, cycle_nodes)
        for idx in xrange(1, len(cycle_nodes)):
            n1 = cycle_nodes[idx - 1]
            n2 = cycle_nodes[idx]
            resid_edge_obj = resid.get_default_weights((n1, n2))

            if back_edges[(n1, n2)]:
                orig_edge_obj = graph.get_default_weights((n2, n1))
                orig_edge_obj[2] = orig_edge_obj[2] - min_cap
            else:
                orig_edge_obj = graph.get_default_weights((n1, n2))
                orig_edge_obj[2] = orig_edge_obj[2] + min_cap

    print result_flow_cost
    return result_flow_cost


def get_flow_cost(graph):
    flow_cost = 0.
    for u in graph.get_nodes():
        for v in graph.get_node_neighbours(u):
            edge_weight_obj = graph.get_default_weights((u, v))
            flow_cost += edge_weight_obj[2] * edge_weight_obj[0]
    return flow_cost


def get_min_resid_cap(resid, cycle_nodes):
    min_cap = float('Inf')
    #if len(cycle_nodes) > 1:
    for i in xrange(1, len(cycle_nodes)):
        n1 = cycle_nodes[i - 1]
        n2 = cycle_nodes[i]
        edge_weight_obj = resid.get_default_weights((n1, n2))
        if edge_weight_obj and edge_weight_obj[1] < min_cap:
            min_cap = edge_weight_obj[1]

    return min_cap


def successive_shortest_path(graph):
    retVal = 0

    working_graph = graph
    #prepare graph and initilize balances
    for edge in working_graph.get_edges():
        if working_graph.get_default_weights(edge)[0] < 0:
            working_graph.get_default_weights(edge)[2] = working_graph.get_default_weights(edge)[1]
            working_graph.get_node_weights(edge[0])[1] += working_graph.get_default_weights(edge)[2]
            working_graph.get_node_weights(edge[1])[1] -= working_graph.get_default_weights(edge)[2]

    while True:
        valid_source = []
        valid_target = []
        equal_nodes = 0
        result_path = None
        resid_graph = None
        pathEdges = []

        for node in working_graph.get_nodes():
            b = working_graph.get_node_weights(node)[0]
            b_prime = working_graph.get_node_weights(node)[1]

            #get valid sourcenodes
            if b - b_prime > 0:
                valid_source.append(node)

            #get valid targetnodes
            if b - b_prime < 0:
                valid_target.append(node)

            #count balanced nodes
            if b == b_prime:
                equal_nodes += 1

        if equal_nodes == working_graph.get_node_count():
            print "Cost Minimal"
            retVal = 1
            break

        if not valid_source or not valid_target:
            print "No Result -> No B-Path"
            break

        #get shortest_path in resid graph
        resid_graph, _ = make_residual_graph_ssp(working_graph)

        for source in valid_source:
            if result_path:
                break
            for target in valid_target:
                result_path = bellman_ford(resid_graph, source, target)
                if result_path:
                    break

        if not result_path or len(result_path) < 2:
            print "No Result -> No B-Path"
            break

        #get gamma
        for index in range(len(result_path) - 1):
            pathEdges.append((result_path[index], result_path[index + 1]))

        minPathCost = min(pathEdges, key=lambda edge: resid_graph.get_default_weights(edge)[1])
        minPathCost = resid_graph.get_default_weights(minPathCost)[1]

        #b(s) - b'(s)
        bS = working_graph.get_node_weights(result_path[0])[0] - working_graph.get_node_weights(result_path[0])[1]
        #b'(t) - b(t)
        bT = working_graph.get_node_weights(result_path[-1])[1] - working_graph.get_node_weights(result_path[-1])[0]
        #determine the minimum here
        gamma = min(minPathCost, bS, bT)

        #update graph
        working_graph = update_graph_from_path_ssp(working_graph, pathEdges, gamma)

    if retVal == 1:
        flow = 0
        for edge in working_graph.get_edges():
            flow += (working_graph.get_default_weights(edge)[2] * working_graph.get_default_weights(edge)[0])
        print flow


def make_residual_graph_ssp(graph):
    back_edges = {}
    resGraph = Graph(directed=True)
    for node in graph.get_nodes():
        resGraph.add_nodes((node, None))

    for edge in graph.get_edges():
        cost = graph.get_default_weights(edge)[0]
        capacity = graph.get_default_weights(edge)[1]
        flow = graph.get_default_weights(edge)[2]
        backEdge = (edge[1], edge[0])

        if flow > 0:
            atr = EdgeProperty(wgt=[-cost, flow])
            resGraph.add_edges([backEdge, atr])
            back_edges[backEdge] = True
        if flow < capacity:
            atr = EdgeProperty(wgt=[cost, capacity - flow])
            resGraph.add_edges([edge, atr])
            back_edges[edge] = False

    return resGraph, back_edges


def update_graph_from_path_ssp(graph, path, gamma):
    result = graph
    for e in path:
        back_e = (e[1], e[0])

        if e in result.get_edges():
            result.get_default_weights(e)[2] += gamma
            result.get_node_weights(e[0])[1] += gamma
            result.get_node_weights(e[1])[1] -= gamma
        elif back_e in result.get_edges():
            result.get_default_weights(back_e)[2] -= gamma
            result.get_node_weights(back_e[0])[1] -= gamma
            result.get_node_weights(back_e[1])[1] += gamma

    return result


def max_matching(graph):
    #add supernodes
    super_source = -1
    super_target = -2

    atr = NodeProperty(wgt=[0, 0])
    graph.add_nodes((super_source, atr))
    graph.add_nodes((super_target, atr))

    #generate b-flow
    added_edges = []
    for node in graph.get_nodes()[:-2]:
        b = graph.get_node_weights(node)[0]

        if b > 0:
            atr = EdgeProperty(wgt=[b, 0])
            graph.add_edges([(super_source, node), atr])
            added_edges.append((super_source, node))
        if b < 0:
            atr = EdgeProperty(wgt=[-b, 0])
            graph.add_edges([(node, super_target), atr])
            added_edges.append((node, super_target))

    #get maxflow
    result_graph = edmonds_karp(graph, super_source, super_target)

    graph.remove_node(super_source)
    graph.remove_node(super_target)

    for edge in added_edges:
        graph.remove_edge(edge)

    #get max matching
    result = []
    for edge in graph.get_edges():
        edge_weight_obj = graph.get_default_weights(edge)
        if edge_weight_obj[1] > 0:
            result.append(edge)

    print result
    print
    print "Max. Matching: %d" % len(result)
    return result
