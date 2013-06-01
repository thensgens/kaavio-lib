from graph import Graph
from basegraph import EdgeProperty, NodeProperty


def convert_matrix(matrix):
    """
        Converts the input adj. matrix
        (it's mandatory that all edges are numbers).
    """
    result_graph = Graph(directed=True)
    parsed_graph = [tuple(entry.strip(' \r\n').split('\t')) for entry in matrix]
    node_count = int(parsed_graph[0][0])
    # remove first item (first item is the node count)
    parsed_graph.pop(0)
    # populate the graph's edge dict
    input_list = [(node, None) for node in range(0, node_count)]
    result_graph.add_nodes(*input_list)

    for line_number, line in enumerate(parsed_graph):
        for column_idx, item in enumerate(line):
            if int(item) == 1:
                result_graph.add_edges([(line_number, column_idx), None])

    # reset the graph to undirected
    result_graph.set_graph_directed(False)
    return result_graph


# TODO: add error handling for invalid documents
def convert_edge_list(edge_list):
    """
        Converts the input edge list
        (it's mandatory that all edges are numbers).
    """
    #result_graph = Graph()
    result_graph = Graph(directed=True)
    parsed_graph = [tuple(entry.strip(' \r\n').split('\t')) for entry in edge_list]
    node_count = int(parsed_graph[0][0])

    # populate the graph's node dict
    input_list = [(node, None) for node in range(0, node_count)]
    result_graph.add_nodes(*input_list)

    # populate the graph's edge dict
    for line in parsed_graph[1:]:
        res_edge, attr = line, None

        if len(line) == 3:
            res_edge = (line[0], line[1])
            attr = EdgeProperty(wgt=[line[2]])

        res_edge = tuple([int(n) for n in res_edge])
        result_graph.add_edges([res_edge, attr])

    return result_graph


def convert_node_edge_list(node_edge_list):
    """
        Converts the input node/edge list
        used in p7 for KostenminimalX.txt
        node = nodeNumber, wgt = [Balance]
        edge = (from, to), wgt = [cost, MaxCapacity, CurrentCapacity]
    """
    result_graph = Graph(directed=True)
    parsed_graph = [tuple(entry.strip(' \r\n').split('\t')) for entry in node_edge_list]
    node_count = int(parsed_graph[0][0])

    # populate the graph's node dict with node_balance
    input_list = []
    for i in range(node_count):
        node_atr = NodeProperty(wgt=[float(parsed_graph[i+1][0])])
        input_list.append((i, node_atr))
    result_graph.add_nodes(*input_list)

    # populate the graph's edge dict
    for line in parsed_graph[node_count+1:]:
        res_edge, attr = line, None

        res_edge = (int(line[0]), int(line[1]))
        #wgt = Cost, MaxCapacity, CurrentCapacity
        attr = EdgeProperty(wgt=[float(line[2]), float(line[3]), 0])

        result_graph.add_edges([res_edge, attr])

    return result_graph


def make_graph_from_mst(mst, input_graph):
    """
        Input: (weight, (u, v))
    """
    res_graph = Graph()

    for node in input_graph.get_nodes():
        res_graph.add_nodes((node, None))

    for w_edge in mst:
        w, u, v = w_edge[0], w_edge[1][0], w_edge[1][1]

        attr = EdgeProperty(wgt=w)
        res_graph.add_edges([(u, v), attr])

    return res_graph
