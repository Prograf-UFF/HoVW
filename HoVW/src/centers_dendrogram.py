import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

def init_args():
    """Input Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="trees' distances matrix")
    parser.add_argument('-c', help="trees' clustered centers")
    parser.add_argument('-o', help="dendogram output path")

    return parser.parse_args()

def main():

    args = init_args()

    A = np.load(args.d)
    C = np.load(args.c)
    output_path = args.o

    centers_distance_matrix = np.zeros((len(C), len(C)), dtype=np.float_)

    for c1, i in zip(C, range(len(C))):
        for c2, j in zip(C, range(len(C))):
            centers_distance_matrix[i][j] = A[c1][c2]

    print(centers_distance_matrix)

    Z = hierarchy.linkage(centers_distance_matrix)

    np.save(output_path + "dendogram.npy", Z)
    
    fig = plt.figure()

    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')

    hierarchy.dendrogram(Z)
    
    plt.savefig(output_path + "dendogram_view.png")
    # plt.show()

main()