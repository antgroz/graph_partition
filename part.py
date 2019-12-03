from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import sys
import numpy as np


def gen_matrix(file_name, mode='A'):

    # Checking input parameters
    if mode not in ['A', 'UNL', 'NL']:
        sys.exit('Error: invalid mode')

    # Opening the file
    txt = open(file_name, 'r', encoding='utf-8')

    # Reading the header in the first line and extracting parameters
    header = txt.readline()
    header = header.rstrip().split(' ')
    vertices_n = int(header[2])
    edges_n = int(header[3])
    clusters_n = int(header[4])

    # Setting up the initial vars
    edges_cnt = 0
    data = []
    row = []
    col = []

    # Starting the cycle to read the file
    while True:
        edge_str = txt.readline()
        if len(edge_str) == 0:
            print('Input file read')
            break
        else:
            edges_cnt += 1
            edge = [int(x) for x in edge_str.rstrip().split(' ')]
            if edge[0] != edge[1]:
                row.extend(edge)
                col.extend(edge[::-1])
                data.extend([1, 1])
                if mode == 'UNL' or mode == 'NL':
                    row.extend(edge)
                    col.extend(edge)
                    data.extend([1, 1])

    # Checking that data was read correctly
    if edges_cnt != edges_n:
        sys.exit('Error: counted number of edges is different from the specified number')

    # Transforming the matrix into an arithmetic-friendly format
    A = coo_matrix((data, (row, col)), shape=(vertices_n, vertices_n), dtype=np.float64)
    A = csr_matrix(A)

    # Normalizing the Laplacian for NL
    if mode == 'NL':
        D = csr_matrix(diags(A.diagonal()**(-1/2)), dtype=np.float64)
        A = D*A*D

    print('Matrix constructed')
    return A, clusters_n


def phi(sparse, labels, k):
    collector = 0
    vertex_list = list(range(sparse.get_shape()[0]))
    for i in range(k):
        partition = [x for x, y in zip(vertex_list, labels) if y == i]
        diff = [x for x in vertex_list if x not in partition]
        connections = csc_matrix(sparse[partition, :])[:, diff]
        collector += connections.sum() / len(partition)
    return collector


if __name__ == '__main__':
    matrix, k = gen_matrix('./graphs_processed/soc-Epinions1.txt', mode='NL')
    w, v = eigsh(matrix, k-1, sigma=0, which='LM', return_eigenvectors=True)
    print('Eigenpairs calculated')

    labels = KMeans(n_clusters=k, max_iter=1000, n_jobs=-1).fit_predict(v)
    # labels = KMeans(n_clusters=k).fit_predict(v)
    # labels = sc.fit_predict(matrix)
    print('Clustering finished')

    print('Phi function: {}'.format(phi(matrix, labels, k)))

    sorted_vert = [vert for label, vert in sorted(zip(labels, list(range(matrix.get_shape()[0]))))]
    matrix = matrix[:, sorted_vert][sorted_vert]
    matrix.setdiag(0)
    ax = plt.gca()
    ax.set_facecolor((0, 0, 0))
    plt.spy(matrix, markersize=1, color=(0, 1, 0))
    plt.show()