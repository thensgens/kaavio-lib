"""
            Mapping   Node -> PropertyObject
        {
            A     : instance_property
            B     : instance_property
            ...
        }

            Mapping   Edge -> PropertyObjects
        {
            (A,B) : [instance_property, ...]
            (B,A) : dict[(1,2)]
            (B,C) : [instance_property, ...]
            ...
        }
"""


class BaseGraph(object):

    """
        Base class for all graph impls.
    """
    def __init__(self):
        self.node_attr = {}
        self.edge_attr = {}

    def get_default_weights(self, edge):
        return self.edge_attr[edge][0].weight

    def get_node_weights(self, node):
        return self.node_attr[node].weight

    def remove_edge(self, edge):
        if self.edge_attr[edge]:
            self.edge_attr.pop(edge)


class BaseProperty(object):

    """
        Base class for all property objects.
    """
    def __init__(self, lbl='', wgt=[]):
        self.label = lbl
        self.weight = wgt


class NodeProperty(BaseProperty):
    pass


class EdgeProperty(BaseProperty):
    pass
