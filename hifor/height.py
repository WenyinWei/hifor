def height_of_node_in_graph(node, graph, write_height_attr=True):
    if graph.out_degree(node) == 0:
        graph.nodes[node]["height"] = 0
        return graph.nodes[node]["height"]
    if "height" in graph.nodes[node]:
        return graph.nodes[node]["height"]
    suc_max_height = 0
    for suc in graph.successors(node):
        suc_max_height = max( 
            suc_max_height, 
            height_of_node_in_graph(suc, graph, write_height_attr=write_height_attr) )
    graph.nodes[node]["height"] = suc_max_height + 1
    return graph.nodes[node]["height"]

def write_heights_of_all_nodes(graph):
    for n in graph.nodes:
        if "height" not in graph.nodes[n]:
            height_of_node_in_graph(n, graph, write_height_attr=True)   

def max_height_of_graph(graph) -> int:
    if len(graph.nodes) == 0:
        raise ValueError("The graph to calculate the maximum of height is empty.")
    write_heights_of_all_nodes(graph)
    max_height = 0
    for n in graph.nodes:
        max_height = max( max_height, graph.nodes[n]["height"] )
    return max_height