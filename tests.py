import sys
from graph import Graph
from basegraph import EdgeProperty
from algorithms import get_coherent_components_count, kruskal, prim, nearest_neighbor, double_tree, brute_force_itertools, start_bnb_bruteforce, dijkstra, bellman_ford, edmonds_karp, make_residual_graph, successive_shortest_path, cycle_cancelling
from utils import convert_matrix, convert_edge_list, convert_node_edge_list
from io import retrieve_information_web, retrieve_information_file


def test_praktikum_1():
    print "=" * 40
    print 'Test from file'
    input_file = 'test_graph.txt'
    result = convert_edge_list(retrieve_information_file(input_file))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    print "=" * 40

    # convert from adj. matrix
    print 'Test from web'
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph1.txt'
    result = convert_matrix(retrieve_information_web(graph_url))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)

    print "=" * 40

    # convert from edge list
    print "Test from web"
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Graph4.txt'
    result = convert_edge_list(retrieve_information_web(graph_url))
    if len(sys.argv) > 1 and sys.argv[1] == 'verbose':
        print result
    print get_coherent_components_count(result)
    print "=" * 40


def test_praktikum_2(arg):
    """
        Reading and converting the graphs (web/file).
    """
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/G_10_20.txt'
    input_file = 'graphs/G_1_2.txt'
    result = convert_edge_list(retrieve_information_file(input_file))
    #result = convert_edge_list(retrieve_information_web(graph_url))

    if arg == 'kruskal':
        """
            Tests for Kruskal
        """
        print "=" * 40
        print "Kruskal algorithm"
        print "=" * 40
        kruskal(result)
    elif arg == 'prim':
        """
            Tests for Prim
        """
        print "=" * 40
        print "Prim algorithm"
        print "=" * 40
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
        print "=" * 40
        print "nearest_neighbor algorithm"
        print "=" * 40
        nearest_neighbor(result, result.get_nodes()[0])

    if arg == 'dt':
        """
            Tests for Kruskal
        """
        print "=" * 40
        print "nearest_neighbor algorithm"
        print "=" * 40
        double_tree(result)


def test_praktikum_4(arg):
    """
        Reading and converting the graphs (web/file).
    """
    # graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/K_10.txt'
    # result = convert_edge_list(retrieve_information_web(graph_url))

    input_file = 'graphs/K_10.txt'
    result = convert_edge_list(retrieve_information_file(input_file))

    if arg == 'bb':
        """
            Tests for Branch-and_Bound
        """
        print "=" * 40
        print "Branch-and-Bound"
        print "=" * 40
        start_bnb_bruteforce(result)

    if arg == 'bf':
        """
            Tests for Brute Force
        """
        print "=" * 40
        print "Brute Force"
        print "=" * 40
        start_bnb_bruteforce(result, False)

    if arg == 'bfiter':
        """
            Tests for Brute-Force (brute-force / itertools)
        """
        print "=" * 40
        print "Brute-Force (itertools.permutations())"
        print "=" * 40
        brute_force_itertools(result)


def test_praktikum_5(arg):
    """
        Reading and converting the graphs (web/file).
    """
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Wege1.txt'
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/G_10_20.txt'
    result = convert_edge_list(retrieve_information_web(graph_url))

    # input_file = 'graphs/K_test.txt'
    # result = convert_edge_list(retrieve_information_file(input_file))

    if arg == 'dij':
        """
            Tests for Dijkstra-Algorithm
        """
        print "=" * 40
        print "Dijkstra-Algorithm"
        print "=" * 40
        try:
            dijkstra(result, int(sys.argv[2]))
        except:
            print 'Usage python tests.py <dij> <source>'

    if arg == 'sp':
        """
            Tests for Shortest-Path (Source -> Target)
        """
        print "=" * 40
        print "Shortest-Path (Source -> Target)"
        print "=" * 40
        try:
            dijkstra(result, int(sys.argv[2]), int(sys.argv[3]))
        except:
            print 'Usage: python tests.py <sp> <start> <target>'

    if arg == 'bell':
        """
            Tests for Bellman-Ford
        """
        print "=" * 40
        print "Bellman-Ford"
        print "=" * 40
        try:
            bellman_ford(result, int(sys.argv[2]))
        except:
            print 'Usage python tests.py <bell> <source>'

def test_praktikum_6(arg):
    """
        Reading and converting the graphs (web/file).
    """
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Fluss.txt'
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/G_1_2.txt'
    readIn = convert_edge_list(retrieve_information_web(graph_url))

    result = Graph(directed=True)
    for node in readIn.get_nodes():
        result.add_nodes((node, None))
    for edge in readIn.get_edges():
        atr = EdgeProperty(wgt=[float(readIn.get_default_weights(edge)[0]), 0])
        result.add_edges([edge, atr])

    # input_file = 'graphs/K_test.txt'
    # result = convert_edge_list(retrieve_information_file(input_file))

    if arg == 'ek':
        """
            Tests for Edmonds-Karp
        """
        print "=" * 40
        print "Edmonds-Karp"
        print "=" * 40
        edmonds_karp(result, int(sys.argv[2]), int(sys.argv[3]))

    if arg == 'resitest':
        """
            Test for Residual Graph Creation
        """
        print "=" * 40
        print "Test for Residual Graph Creation"
        print "=" * 40

        make_residual_graph(result)

def test_praktikum_7(arg):
    """
        Reading and converting the graphs (web/file).
    """
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Kostenminimal5.txt'
    graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Kostenminimal100_1.txt'
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Kostenminimal100_2.txt'
    #graph_url = 'http://www.hoever.fh-aachen.de/webDateien/mmi/Grafen/Kostenminimal100_3.txt'
    result = convert_node_edge_list(retrieve_information_web(graph_url))

    # input_file = 'graphs/K_test.txt'
    # result = convert_edge_list(retrieve_information_file(input_file))

    if arg == 'cc':
        """
            Tests for Cycle-Cancelling
        """
        print "=" * 40
        print "Cycle-Cancelling"
        print "=" * 40
        cycle_cancelling(result)

    if arg == 'ssp':
        """
            Test for Successive Shortest-Path
        """
        print "=" * 40
        print "Test for Successive Shortest-Path"
        print "=" * 40
        successive_shortest_path(result)

if __name__ == '__main__':
    """
        sys.argv[1] should contain the specified algorithm, e.g. 'kruskal'.
        sys.argv[1+n] should contain additional arguments, e.g. start node for dijkstra/bellman-ford.
    """
    try:
        arg = sys.argv[1]
    except:
        arg = None

    #test_praktikum_1()
    #test_praktikum_2(arg)
    #test_praktikum_3(arg)
    #test_praktikum_4(arg)
    #test_praktikum_5(arg)
    #test_praktikum_6(arg)
    test_praktikum_7(arg)
