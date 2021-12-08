from .height import max_height_of_graph
from sympy import latex

def recursive_draw_nodes_from_root(
    graph, root, fig, ax, level_indent, level_texts, 
    gap_between_terms=0.3, box_pad = 0.4, line_gap = 1.5, fontsize=24):
    
    if "Eq" in graph.nodes[root]:
        text = ax.text(
            level_indent[graph.nodes[root]["height"]], 
            graph.nodes[root]["height"] * line_gap, 
            r"$" + latex(graph.nodes[root]["Eq"], mode="inline") + r"$", fontsize=fontsize,
            bbox = dict(facecolor='none', edgecolor='black', boxstyle=f'round, pad={box_pad}'))
    else:
        text = ax.text(
            level_indent[graph.nodes[root]["height"]], 
            graph.nodes[root]["height"] * line_gap, 
            r"$" + latex(root, mode="inline") + r"$", fontsize=fontsize,
            bbox = dict(facecolor='none', edgecolor='black', boxstyle=f'round, pad={box_pad}'))
    level_texts[ graph.nodes[root]["height"] ].append(text)
    fig.canvas.draw()
    bbox = fig.gca().transData.inverted().transform_bbox(text.get_window_extent())
    level_indent[graph.nodes[root]["height"]] += bbox.x1 - bbox.x0 + gap_between_terms
    for h in range(graph.nodes[root]["height"]+1, max_height_of_graph(graph)+1 ):
        level_indent[h] = max(level_indent[h], level_indent[graph.nodes[root]["height"]])
        
    
    if graph.nodes[root]["height"] != 0:
        for suc in graph.successors(root):
            recursive_draw_nodes_from_root(
                graph, suc, fig, ax, level_indent, level_texts, 
                gap_between_terms, box_pad, line_gap, fontsize)
            suc_text = level_texts[ graph.nodes[suc]["height"] ][-1]
            fig.canvas.draw()
            suc_bbox = fig.gca().transData.inverted().transform_bbox(suc_text.get_window_extent())
            # refer to Matplotlib doc for details about the arrow properties
            # https://matplotlib.org/stable/tutorials/text/annotations.html#annotating-with-arrow
            ax.annotate("",
               xy=( (bbox.x0+bbox.x1)/2, bbox.y0 - line_gap * 0.4  ), xycoords='data',
               xytext=( (suc_bbox.x0+suc_bbox.x1)/2, suc_bbox.y1 + box_pad/4 ), textcoords='data',
               arrowprops=dict(arrowstyle="<-", connectionstyle="angle, angleA=-90,angleB=180,rad=5"))
        ax.annotate("",
           xy=( (bbox.x0+bbox.x1)/2, bbox.y0 - box_pad/4  ), xycoords='data',
           xytext=( (bbox.x0+bbox.x1)/2, bbox.y0 - line_gap * 0.4  ), textcoords='data',
           arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"))

def draw_formula_hierarchy(
    graph, fig, ax, **kwarg):
    level_indent = [0.0] * (max_height_of_graph(graph)+1)
    level_texts = [ [] for _ in range(max_height_of_graph(graph)+1) ] 
    for root in graph.nodes:
        if graph.in_degree(root) == 0:
            recursive_draw_nodes_from_root(
                graph, root, fig, ax, level_indent, level_texts, **kwarg)