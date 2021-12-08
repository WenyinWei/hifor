from networkx import is_directed_acyclic_graph

def add_nodes_and_edges_from_Eq(graph, eq):
    graph.add_nodes_from([
        (eq.lhs, {"Eq": eq}),
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

def lambdify_graph(graph):
    if not is_single_source_DAG(graph):
        raise ValueError("The graph to be lambdified must be a single source directed acyclic graph.")
    else:
        _source_set = source_set_of_DAG(graph)
    _source_set = 
    
    return 