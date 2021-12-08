
from .height import write_heights_of_all_nodes, max_height_of_graph
from sympy import latex
import matplotlib.pyplot as plt
import sympy # for sympy.Integer

def recursive_draw_nodes_from_root(
    graph, root, fig, ax, level_indent, level_texts, 
    equal_list=["expr", "val"], gap_between_terms=0.3, box_pad = 0.4, line_gap = 1.5, fontsize=24):
    
    text_string = r"$$" + latex(root) 
    for term in equal_list:
        if term in graph.nodes[root].keys():
            if term=="expr":
                text_string += "=" + latex(graph.nodes[root][term]) 
            elif term=="val":
                if isinstance(graph.nodes[root][term]["value"], int ) or isinstance(graph.nodes[root][term]["value"], sympy.Integer ):
                    text_string += "= " + str(graph.nodes[root][term]["value"]) + graph.nodes[root][term]["unit"]
                else:
                    st = "{:0.4e}".format(graph.nodes[root][term]["value"])
                    deci_part, exp_part = st.split("e")
                    if exp_part=="+0":
                        text_string += f"= {deci_part}" + graph.nodes[root][term]["unit"]
                    else:
                        text_string += f"= {deci_part}*10^{{ {exp_part} }} " + graph.nodes[root][term]["unit"]
    text_string += r"$$"
    text = ax.text(
        level_indent[graph.nodes[root]["height"]], 
        graph.nodes[root]["height"] * line_gap, 
        text_string, fontsize=fontsize,
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
                equal_list, gap_between_terms, box_pad, line_gap, fontsize)
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

def draw_hifor(graph, xlim=None, ylim=None, **kwarg):
    write_heights_of_all_nodes(graph)
    plt.rcParams['text.usetex'] = True

    if ylim is None:
        if "line_gap" in kwarg.keys():
            ylim = [0.0, max_height_of_graph(graph) * kwarg["line_gap"] + 0.5 ]
        else:
            ylim = [0.0, max_height_of_graph(graph) * 1.5               + 0.5 ]
    
    figsize_data_ratio = 2.3
    if xlim is None:
        xlim_try = [0, 5]
        fig, ax = plt.subplots(1,1, figsize=(
            figsize_data_ratio * (xlim_try[1] - xlim_try[0]), 
            figsize_data_ratio * (ylim[1] - ylim[0])  ) )
        ax.set_xlim(xlim_try); ax.set_ylim(ylim)
    else:
        fig, ax = plt.subplots(1,1, figsize=(
            figsize_data_ratio * (xlim[1] - xlim[0]), 
            figsize_data_ratio * (ylim[1] - ylim[0]) ) )
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_axis_off()

    level_indent = [0.0] * (max_height_of_graph(graph)+1)
    level_texts = [ [] for _ in range(max_height_of_graph(graph)+1) ] 
    for root in graph.nodes:
        if graph.in_degree(root) == 0:
            recursive_draw_nodes_from_root(
                graph, root, fig, ax, level_indent, level_texts, **kwarg)
    if xlim is None:
        plt.close(fig)
        return draw_hifor(graph, xlim=[0.0, level_indent[-1]], ylim=ylim, **kwarg)

    return fig, ax