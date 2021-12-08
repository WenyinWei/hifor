def calc_value_of_node(graph, node, write_on_graph:bool=True):
    """calc the value of a single node in the specified directed acyclic graph (hierarchy of formulas)

    Args:
        graph (DAG): the graph to be calculated every node value. 
        write_on_graph (bool, optional): [description]. Defaults to True.
    
    Note:
        This function does overwrite the existing values.
    """
    if graph.out_degree(node)==0:
        if "val" in graph.nodes[node].keys():
            return graph.nodes[node]["val"]["value"]
        else:
            raise ValueError(f"The node {node} lacks value, which causes the failure of calculation.")
    
    result = graph.nodes[node]["expr"].evalf( subs={
        suc: calc_value_of_node(graph, suc) for suc in graph.successors(node)
    })

    if write_on_graph:
        if "val" in graph.nodes[node]:
            graph.nodes[node]["val"] = {"value": result, "unit": graph.nodes[node]["val"]["unit"]}
        else:
            graph.nodes[node]["val"] = {"value": result, "unit": "[?]"}
    return result


def calc_values_of_whole_graph(graph):
    """calc the empty values of the whole directed acyclic graph (hierarchy of formulas)

    Args:
        graph (DAG): the graph to be calculated every node value. 
        write_on_graph (bool, optional): [description]. Defaults to True.
    
    Note:
        This function does not overwrite the existing values.
    """
    for n in graph.nodes:
        if not "val" in graph.nodes[n]:
            calc_value_of_node(graph, n, write_on_graph=True)

# def modify_value_of_node(graph, node, val):
#     graph.nodes[node]["val"] = val
    
#     for pre in graph.predecessors(node):
#         modify_value_of_node