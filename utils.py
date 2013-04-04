import urllib2, sys
from graph import Graph
from basegraph import EdgeProperty
from algorithms import (
        recursive_depth_first_search,
        iterative_breadth_first_search,
        get_coherent_components_count)

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
    result_graph = Graph()
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


def retrieve_information_web(url):
    try:
        res = urllib2.urlopen(url).readlines()
    except:
        res = []
    return res


def retrieve_information_file(input):
    res = []
    with open(input, "r") as f:
        res = f.readlines()
    return res


if __name__ == '__main__':
    print "Test from file"
    input_file = "test_graph.txt"
    result = convert_edge_list(retrieve_information_file(input_file))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    print "=" * 30

    # convert from adj. matrix
    print "Test from web"
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph1.txt'
    result = convert_matrix(retrieve_information_web(graph_url))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    # convert from edge list
    #print "Test from web"
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph1.txt'
    #result = convert_edge_list(retrieve_information_web(graph_url), True)
    #if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        #print result
    #print get_coherent_components_count(result)
