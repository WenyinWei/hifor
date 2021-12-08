from hifor.height import write_heights_of_all_nodes
from networkx import is_directed_acyclic_graph
from sympy import Eq


def add_nodes_and_edges_from_Eq(graph, eq):
    graph.add_nodes_from([
        (eq.lhs, {"expr": eq.rhs}),
        *eq.rhs.free_symbols
    ])
    for free_sym in eq.rhs.free_symbols:
        graph.add_edge(eq.lhs, free_sym)
    

def is_single_source_DAG(graph):
    if not is_directed_acyclic_graph(graph):
        raise ValueError("The graph has to be a directed acyclic graph first before it is a single-source DAG.")
    _source_num = 0
    for n in graph.nodes:
        if graph.in_degree(n) == 0:
            _source_num += 1
        if graph.in_degree(n) + graph.out_degree(n) == 0:
            raise ValueError("There exists an isolated node in the graph, we do not check the single-source property of such a graph.")
    if _source_num == 1:
        return True
    else:
        return False
def source_set_of_DAG(graph):
    _sources = set()
    for n in graph.nodes:
        if graph.in_degree(n) == 0:
            _sources.add(n)
    return _sources

def is_single_sink_DAG(graph):
    if not is_directed_acyclic_graph(graph):
        raise ValueError("The graph has to be a directed acyclic graph first before it is a single-sink DAG.")
    _sink_num = 0
    for n in graph.nodes:
        if graph.out_degree(n) == 0:
            _sink_num += 1
        if graph.in_degree(n) + graph.out_degree(n) == 0:
            raise ValueError("There exists an isolated node in the graph, we do not check the single-sink property of such a graph.")
    if _sink_num == 1:
        return True
    else:
        return False
def sink_set_of_DAG(graph):
    _sinks = set()
    for n in graph.nodes:
        if graph.out_degree(n) == 0:
            _sinks.add(n)
    return _sinks

def condensate_hifor_to_root_and_its_expr(graph):
    if not is_single_source_DAG(graph):
        raise ValueError("The graph to be lambdified must be a single source directed acyclic graph.")
    else:
        _source_set = source_set_of_DAG(graph)
    write_heights_of_all_nodes(graph)
    for _root in _source_set: # in fact there is only one source
        _expr = graph.nodes[_root]["expr"]
        while any(list( map(lambda x: graph.nodes[x]["height"]>0, _expr.free_symbols) )):
            for free_sym in _expr.free_symbols:
                if graph.nodes[free_sym]["height"] > 0:
                    _expr = _expr.subs(free_sym, graph.nodes[free_sym]["expr"])
    
    return _root, _expr

def embed_subgraph_into_graph(subgraph, graph, created_var_in_graph, expr_arg_subs_dict):
    subgraph_root, subgraph_expr = condensate_hifor_to_root_and_its_expr(subgraph)
    _parameter_set = subgraph_expr.free_symbols - expr_arg_subs_dict.keys()
    for n in _parameter_set:
        if not "val" in subgraph.nodes[n]:    
            raise ValueError(f"The {n} free symbol of subgraph cannot find the correspondence in the specified `eqarg_subs_dict`.")
    # module_node = (subgraph_root, subgraph_expr, expr_arg_subs_dict) # The subgraph is now regarded as a module embedded in the graph.
    graph.add_node(created_var_in_graph, expr=subgraph_expr.subs(expr_arg_subs_dict) )
    for param_sym in _parameter_set:
        graph.add_node(param_sym, **subgraph.nodes[param_sym] )
    for free_sym in subgraph_expr.free_symbols:
        if free_sym in _parameter_set:
            graph.add_edge(created_var_in_graph, free_sym)
        else:
            graph.add_edge(created_var_in_graph, expr_arg_subs_dict[free_sym])
    