from graphviz import Digraph

def trace(root):
    """enumerate through graphs and edges"""
    # build set of nodes and edges
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, filename='graph', save_as_png=True):
    """visualize nodes in the forward pass and optionally save as PNG"""
    dot = Digraph(format='png', filename=filename, graph_attr={'rankdir': 'LR'})  # left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')  # create a node object
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)  # create an edge object
    for n1, n2 in edges:
        # connect nodes
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    if save_as_png:
        dot.render()  # Render and save the graph as a PNG file
    
    return dot
