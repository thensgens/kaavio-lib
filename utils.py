import urllib2
from graph import Graph

def convert_matrix(matrix):
    pass


def convert_edge_list(raw_edge_list):
    """
        Converts the input edge list
        (it's mandatory that all edges are numbers).
        Input parameter is the raw input.
    """
    result_graph = Graph()



def retrieve_information_web(url):
    try:
        res = urllib2.urlopen(url).readlines()
    except:
        res = []
    return res


def retrieve_information_file():
    pass
