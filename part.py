from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import sys
import numpy as np
# import pyamg


def gen_matrix(file_name, mode='A', adj=False):

    # Checking input parameters
    if mode not in ['A', 'UNL', 'NL']:
        sys.exit('Error: invalid mode')

    # Opening the file
    txt = open(file_name, 'r', encoding='utf-8')

    # Reading the header in the first line and extracting parameters
    header_str = txt.readline()
    header = header_str.rstrip().split(' ')
    vertices_n = int(header[2])
    edges_n = int(header[3])
    clusters_n = int(header[4])

    # Setting up the initial vars
    edges_cnt = 0
    data = []
    row = []
    col = []
    base = 1 if mode == 'A' else -1
    edges = []

    # Starting the cycle to read the file
    while True:
        edge_str = txt.readline()
        if len(edge_str) == 0:
            print('Input file read')
            break
        else:
            edges_cnt += 1
            edge = [int(x) for x in edge_str.rstrip().split(' ')]
            edges.append(edge)
            if edge[0] != edge[1]:
                row.extend(edge)
                col.extend(edge[::-1])
                data.extend([base, base])
                if mode != 'A':
                    row.extend(edge)
                    col.extend(edge)
                    data.extend([1, 1])

    # Checking that data was read correctly
    if edges_cnt != edges_n:
        sys.exit('Error: counted number of edges is different from the specified number')

    # Transforming the matrix into an arithmetic-friendly format
    A = coo_matrix((data, (row, col)), shape=(vertices_n, vertices_n), dtype=np.float64)
    A = csr_matrix(A)

    # Getting the adjacency matrix
    if mode != 'A' and adj is True:
        adjacency = -A
        adjacency.setdiag(0)

    # Normalizing the Laplacian for NL
    if mode == 'NL':
        D = csr_matrix(diags(A.diagonal()**(-1/2)), dtype=np.float64)
        A = D*A*D

    print('Matrix constructed')
    if mode != 'A' and adj is True:
        return A, clusters_n, header_str, edges, adjacency
    else:
        return A, clusters_n, header_str, edges


def phi(edges, labels, k):
    cnt = np.zeros(k)
    clr = np.zeros(k)

    for i in range(len(labels)):
        cnt[labels[i]] += 1

    for i in range(len(edges)):
        label0 = labels[edges[i][0]]
        label1 = labels[edges[i][1]]
        if label0 != label1:
            clr[label0] += 1
            clr[label1] += 1

    return np.dot(clr, cnt**(-1))


def output(header, graph_name, vertices_n, labels, num_vect):
    with open('{}_{}.output'.format(graph_name, str(num_vect)), 'w', encoding='utf-8') as f:
        f.write(header)
        for i in range(vertices_n):
            f.write('{} {}\n'.format(i, labels[i]))


if __name__ == '__main__':
    graph_name = sys.argv[1]
    param = float(sys.argv[2])
    matrix, k, header, edges, adjacency = gen_matrix('./graphs_processed/{}.txt'.format(graph_name), mode='UNL', adj=True)
    n = matrix.get_shape()[0]
    w, v = eigsh(matrix, int(param*k), sigma=0, which='LM', return_eigenvectors=True, tol=1e-4)
    print('Eigenpairs found')

    labels = KMeans(n_clusters=k, n_jobs=-1).fit_predict(v)
    # labels = KMeans(n_clusters=k).fit_predict(v)
    # labels = sc.fit_predict(matrix)
    print('Clustering finished')

    # Writing the results
    output(header, graph_name, n, labels, int(param*k))
    print('Results recorded')

    print('Phi function: {}'.format(phi(edges, labels, k)))

    # sorted_vert = [vert for label, vert in sorted(zip(labels, list(range(n))))]
    # adjacency = adjacency[:, sorted_vert][sorted_vert]
    # adjacency.setdiag(0)
    # ax = plt.gca()
    # ax.set_facecolor((0, 0, 0))
    # plt.spy(adjacency, markersize=1, color=(0, 1, 0))
    # plt.show()