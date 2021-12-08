def calc_value_of_node(graph, node, write_on_graph=True):
    if graph.out_degree(node)==0:
        if "val" in graph.nodes[node].keys():
            return graph.nodes[node]["val"]["value"]
        else:
            raise ValueError(f"The node {node} lacks value, which causes the failure of calculation.")
    
    result = graph.nodes[node]["expr"].subs({
        suc: calc_value_of_node(graph, suc) for suc in graph.successors(node)
    })

    if write_on_graph:
        if "val" in graph.nodes[node]:
            graph.nodes[node]["val"] = {"value": result, "unit": graph.nodes[node]["unit"]}
        else:
            graph.nodes[node]["val"] = {"value": result, "unit": "[?]"}
    return result

# def modify_value_of_node(graph, node, val):
#     graph.nodes[node]["val"] = val
    
#     for pre in graph.predecessors(node):
#         modify_value_of_node