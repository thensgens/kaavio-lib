from basegraph import BaseGraph
from graph_exceptions import NodeNotInGraph


class Graph(BaseGraph):

    IS_GRAPH_DIRECTED = False

    def __init__(self):
        BaseGraph.__init__(self)
        """
            Example for the adjacency list structure:
                'A' -> ['B', 'C'],
                'B' -> ['A', 'D']
        """
        self.node_adj_list = {}

    def add_nodes(self, *nodes):
        for node_tuple in nodes:
            node, node_attribute = node_tuple[0], node_tuple[1]
            if node not in self.node_adj_list:
                self.node_adj_list[node] = []
                self.node_attr[node] = node_attribute

    def add_edges(self, *edges):
        for edge_list in edges:
            edge, edge_attribute = edge_list[0], edge_list[1]
            self.add_node_neighbours(edge)
            self.add_edge_attributes(edge, edge_attribute)

    def add_node_neighbours(self, edge):
        src, dest = edge[0], edge[1]
        if src not in self.node_adj_list or dest not in self.node_adj_list:
            raise NodeNotInGraph()
        else:
            self.node_adj_list[src].append(dest)
            if not self.__graph_is_directed():
                self.node_adj_list[dest].append(src)

    def add_edge_attributes(self, edge, edge_attribute):
        self.edge_attr.setdefault(edge, []).append(edge_attribute)
        if not self.__graph_is_directed():
            new_edge = (edge[1], edge[0])
            self.edge_attr.setdefault(new_edge, []).append(edge_attribute)

    def __graph_is_directed(self):
        return self.IS_GRAPH_DIRECTED

    def __repr__(self):
        #return ' / '.join(self.node_adj_list)
        for node, neighbours in self.node_adj_list.items():
            print '%s  ->  %s' % (node, neighbours)

if __name__ == '__main__':
    gr = Graph()
    gr.add_nodes(('A', None), ('B', None), ('C', None))
    gr.add_edges([('A', 'B'), None])
    gr.add_edges([('A', 'B'), None], [('B', 'C'), None])
    print gr
