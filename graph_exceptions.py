class NodeNotInGraph(RuntimeError):
    """
        This exception occurs if an edge should be added,
        but the corresponding nodes were not added yet.
    """
    pass
