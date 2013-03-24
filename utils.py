import urllib2
from graph import Graph
from basegraph import BaseProperty


def convert_matrix(matrix):
    pass


def convert_edge_list(raw_edge_list):
    """
        Converts the input edge list
        (it's mandatory that all edges are numbers).
        Input parameter is the raw input.
    """
    result_graph = Graph()
    parsed_graph = [ tuple(entry.strip(' \r\n').split('\t')) for entry in raw_edge_list ]
    node_count = int(parsed_graph[0][0])

    # populate the graph's node dict
    input_list = [(node, None) for node in range(0, node_count)]
    result_graph.add_nodes(*input_list)

    # populate the graph's edge dict
    for line in parsed_graph[1:]:
        res_edge, attr = line, None

        if len(line) == 3:
            res_edge = (line[0], line[1])
            attr = BaseProperty(wgt=[line[2]])

        res_edge = tuple([int(n) for n in res_edge])
        result_graph.add_edges([res_edge, attr])

    print result_graph

def retrieve_information_web(url):
    try:
        res = urllib2.urlopen(url).readlines()
    except:
        res = []
    return res


def retrieve_information_file():
    pass





if __name__ == '__main__':
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph2.txt'
    result = convert_edge_list(retrieve_information_web(graph_url))
