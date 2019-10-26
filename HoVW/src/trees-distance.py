import pickle, argparse, collections, time, multiprocessing, math
from itertools import product, combinations, islice
from functools import partial
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np
import pickle4reducer 
from scipy.spatial.distance import cdist
from apted import APTED, Config, PerEditOperationConfig
from apted.helpers import Tree
from image import Image

class ImageTreeDistance(Config):
    def __init__(self, centroids_matrix):
        self.centroids_matrix = centroids_matrix
        self.centroids_mean = centroids_matrix.mean()

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        #a = np.max([(node.neighbors), (node.depth)])
        a = np.max([np.log2(node.neighbors), np.log2(node.depth)])
        a = 1/a if a > 0 else 0
        return self.centroids_mean * a

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        return self.delete(node)
    
    def rename(self, node1, node2):
        """Compares attribute .value of trees"""
        # ATTENTION: if the node is a root node and if try to relabeled
        # it the cost is zero
        if node1.label == None or node2.label == None:
            return 0 
            
        return self.centroids_matrix[node1.label][node2.label]
    
    def children(self, node):
        return node.childs

def init_args():
    """Input Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='images tree labeled file')
    parser.add_argument('-c', help='clusters distance file')
    parser.add_argument('-m', help='output trees distance matrix')
    parser.add_argument('-d', help='apted custom tree edit distance class')
    parser.add_argument('-p', help='qtd parallel process')

    return parser.parse_args()

def heatmap_plot(distances):
    plt.imshow(distances, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.title("Hierarchies Distance Matrix Heatmap")
    plt.savefig("distance_heatmap.png")

def calc_dist_trees(t1, t2, clusters_dist):
    print(t1.tree.name, t2.tree.name)
    return APTED(t1.tree.root, t2.tree.root, 
        ImageTreeDistance(clusters_dist)).compute_edit_distance()

def combinations_matrix(shape, combinations_list):
    A = np.zeros(shape, dtype=np.float_)
    limit = combinations_list.shape[0]-1
    b0 = 0
    for a,j in zip(A, range(A.shape[1]-1)):
        t = a.shape[0] - (j+1)
        b1 = b0 + t
        b1 = b1 if b1 < limit else limit
        if(b0 == b1): b1 += 1
        assign = combinations_list[b0:b1]
        a[j+1:] = assign
        A[j+1:,j] = assign
        b0 = b1

    return A

def main():
    args = init_args()

    trees_path = args.i
    cdist_path = args.c
    apted_dist_save = args.d
    tmatrix_path = args.m
    qtd_process = int(args.p)

    img_trees = []
    with open(trees_path, 'rb') as f:
        while True:
            try:
                img_trees.append(pickle.load(f))
            except EOFError:
                break

    a = clusters_dist = np.load(cdist_path)

    i, j = np.argwhere(a == np.min(a[np.where(a > 0)]))[0]
    print("Min = A[%d,%d] = %.16f" % (i, j, a[i,j]))

    i,j = np.unravel_index(a.argmax(), a.shape)
    print("Max = A[%d,%d] = %.16f" % (i, j, a[i,j]))

    print("Mean(A) =", a.mean(), "\nMedian(A) =", np.median(a))

    with open(path.join(apted_dist_save), 'wb') as handle:
        pickle.dump(ImageTreeDistance(clusters_dist), handle, protocol=pickle.HIGHEST_PROTOCOL)

    calc_dist_trees_partial = partial(calc_dist_trees,clusters_dist=clusters_dist)

    #parallel
    t0 = time.time()
    
    repetition = 2
    comb_iter = combinations(img_trees, r=repetition)

    """
        n!/(r!*(n-r)!) = 1/r! * n * (n-1) * ... * (n-r+1)
    """
    comb_iter_size = len(img_trees)*(len(img_trees)-1)//math.factorial(repetition)

    tree_distances = []
    chuncks = 1
 
    # ctx = multiprocessing.get_context()
    # ctx.reducer = pickle4reducer.Pickle4Reducer()

    for i in range(chuncks):
        print("Chunck {}".format(i))
        comb_sliced = islice(comb_iter, 0, comb_iter_size//chuncks)
        with multiprocessing.Pool(processes=qtd_process) as pool:
            td = pool.starmap(calc_dist_trees_partial, comb_sliced)
        tree_distances += td
    
    tree_distances = combinations_matrix((len(img_trees),len(img_trees)), np.array(tree_distances))


    print("Parallel Time: ", time.time() - t0)

    # #sequential
    # tree_distances = np.zeros((len(img_trees), len(img_trees)), np.float_)
    # t0 = time.time()

    # for t1, i in zip(img_trees, range(len(img_trees))):
    #     print(i)
    #     tree_distances[i] = ([calc_dist_trees_partial(t1, t2) for t2 in img_trees])

    # print("Sequential Time: ", time.time() - t0)
    
    print(tree_distances)
    np.save(tmatrix_path, tree_distances)

    heatmap_plot(tree_distances)

if __name__ == "__main__":
    main()