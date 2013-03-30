import urllib2
from graph import Graph
from basegraph import EdgeProperty
from algorithms import recursive_depth_first_search
from algorithms import iterative_breadth_first_search

def convert_matrix(matrix):
    pass


# TODO: add error handling for invalid documents
def convert_edge_list(edge_list, raw=False):
    """
        Converts the input edge list
        (it's mandatory that all edges are numbers).
        Input parameter is the raw input.
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
    """
    # Test from web
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph2.txt'
    result = convert_edge_list(retrieve_information_web(graph_url), True)
    #print recursive_depth_first_search(result, 0)    
    print iterative_breadth_first_search(result, 0)
    """

    # Test from file
    input_file = "test_graph.txt"
    result = convert_edge_list(retrieve_information_file(input_file), False)
    print recursive_depth_first_search(result, 0)
    print "=========================="
    print iterative_breadth_first_search(result, 0)
