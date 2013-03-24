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


class BaseProperty(object):

    """
        Base class for all property objects.
    """
    def __init__(self, lbl='', wgt=[]):
        self.label = lbl
        self.weight = wgt

if __name__ == '__main__':
    b = BaseGraph()
