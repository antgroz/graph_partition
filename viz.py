from part import gen_matrix
import networkx as nx
import matplotlib.pyplot as plt


def save_graph(graph, file_name):
    plt.figure(num=None, figsize=(20, 20))
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    del fig


if __name__ == "__main__":
    matrix, k = gen_matrix('Oregon-1.txt', 'A')
    G = nx.convert_matrix.from_scipy_sparse_matrix(matrix)
    save_graph(G, 'graph.jpg')