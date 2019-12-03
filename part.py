from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import sys
import numpy as np
import cupy as cp


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
    # while True:
    #     edge_str = txt.readline()
    #     if len(edge_str) == 0:
    #         print('Input file read')
    #         break
    #     else:
    #         edges_cnt += 1
    #         edge = [int(x) for x in edge_str.rstrip().split(' ')]
    #         if edge[0] != edge[1]:
    #             row.extend(edge)
    #             col.extend(edge[::-1])
    #             data.extend([1, 1])
    #             if mode == 'UNL' or mode == 'NL':
    #                 row.extend(edge)
    #                 col.extend(edge)
    #                 data.extend([1, 1])

    A = cp.ndarray((vertices_n, vertices_n), dtype=cp.float64)

    # Reading the file into a CuPy array
    while True:
        edge_str = txt.readline()
        if len(edge_str) == 0:
            print('Input file read')
            break
        else:
            edges_cnt += 1
            edge = [int(x) for x in edge_str.rstrip().split(' ')]
            if edge[0] != edge[1] and A[edge[0], edge[1]] != 1:
                A[edge[0], edge[1]] = 1
                A[edge[1], edge[0]] = 1
                if mode == 'UNL':
                    A[edge[0], edge[0]] += 1
                    A[edge[1], edge[1]] += 1

    # Checking that data was read correctly
    if edges_cnt != edges_n:
        sys.exit('Error: counted number of edges is different from the specified number')

    # Transforming the matrix into an arithmetic-friendly format
    # A = coo_matrix((data, (row, col)), shape=(vertices_n, vertices_n), dtype=np.float64)
    # # A = csr_matrix(A)
    # A = cp.ndarray(A)

    # Normalizing the Laplacian for NL
    if mode == 'NL':
        D = csr_matrix(diags(A.diagonal()**(-1/2)), dtype=np.float64)
        A = D*A*D
        for vertex in range(vertices_n):
            A[vertex, vertex] = 1.0

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
    matrix, k = gen_matrix('ca-GrQc.txt', mode='A')
    # w, v = eigsh(matrix, 4, sigma=0, which='LM', return_eigenvectors=True)
    # labels = MiniBatchKMeans(n_clusters=k).fit_predict(v)
    # labels = KMeans(n_clusters=k).fit_predict(v)
    sc = SpectralClustering(k, affinity='precomputed', n_init=500, assign_labels='discretize')
    labels = sc.fit_predict(matrix)
    print(phi(matrix, labels, k))

    sorted_vert = [vert for label, vert in sorted(zip(labels, list(range(matrix.get_shape()[0]))))]
    matrix = matrix[:, sorted_vert][sorted_vert]
    matrix.setdiag(0)
    ax = plt.gca()
    ax.set_facecolor((0, 0, 0))
    plt.spy(matrix, markersize=1, color=(0, 1, 0))
    plt.show()