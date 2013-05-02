import sys
from algorithms import get_coherent_components_count, kruskal, prim, nearest_neighbor, double_tree, branch_and_bound, branch_bound_backtrack_start
from utils import convert_matrix, convert_edge_list
from io import retrieve_information_web, retrieve_information_file


def test_praktikum_1():
    print "=" * 30
    print 'Test from file'
    input_file = 'test_graph.txt'
    result = convert_edge_list(retrieve_information_file(input_file))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    print "=" * 30

    # convert from adj. matrix
    print 'Test from web'
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph1.txt'
    result = convert_matrix(retrieve_information_web(graph_url))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    print "=" * 30

    # convert from edge list
    print "Test from web"
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph4.txt'
    result = convert_edge_list(retrieve_information_web(graph_url))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)
    print "=" * 30


def test_praktikum_2(arg):
    """
        Reading and converting the graphs (web/file).
    """
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/G_10_20.txt'
    input_file = 'graphs/G_10_200.txt'
    result = convert_edge_list(retrieve_information_file(input_file))
    #result = convert_edge_list(retrieve_information_web(graph_url))

    if arg == 'kruskal':
        """
            Tests for Kruskal
        """
        print "=" * 30
        print "Kruskal algorithm"
        print "=" * 30
        kruskal(result)
    elif arg == 'prim':
        """
            Tests for Prim
        """
        print "=" * 30
        print "Prim algorithm"
        print "=" * 30
        prim(result, result.get_nodes()[0])


def test_praktikum_3(arg):
    """
        Reading and converting the graphs (web/file).
    """
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/G_10_20.txt'
    input_file = 'graphs/K_10.txt'
    result = convert_edge_list(retrieve_information_file(input_file))
    #result = convert_edge_list(retrieve_information_web(graph_url))

    if arg == 'nn':
        """
            Tests for Kruskal
        """
        print "=" * 30
        print "nearest_neighbor algorithm"
        print "=" * 30
        nearest_neighbor(result, result.get_nodes()[0])

    if arg == 'dt':
        """
            Tests for Kruskal
        """
        print "=" * 30
        print "nearest_neighbor algorithm"
        print "=" * 30
        double_tree(result)

def test_praktikum_4(arg):
    """
        Reading and converting the graphs (web/file).
    """
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/K_10.txt'
    #input_file = 'graphs/K_10.txt'
    result = convert_edge_list(retrieve_information_web(graph_url))
    #result = convert_edge_list(retrieve_information_web(graph_url))

    if arg == 'bb':
        """
            Tests for Branch-and_Bound
        """
        print "=" * 30
        print "Branch-and-Bound"
        print "=" * 30
        #branch_and_bound(result)
        branch_bound_backtrack_start(result)


if __name__ == '__main__':
    """
        param arg contains the specified algorithm, e.g. 'kruskal'
    """
    try:
        arg = sys.argv[1]
    except:
        arg = None

    #test_praktikum_1()
    #test_praktikum_2(arg)
    #test_praktikum_3(arg)
    test_praktikum_4(arg)
