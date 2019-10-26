#!/usr/bin/env python

import pickle, argparse
import numpy as np
from apted import APTED, Config
from utils import Log
from image import Image
from clusters import Cluster


# TODO: Mesma classe que em trees-distance; precisa colocar em um .py separado
class ImageTreeDistance(Config):
    def __init__(self, centroids_matrix):
        self.centroids_matrix = centroids_matrix
        self.centroids_mean = centroids_matrix.mean()

    def delete(self, node):
        """Calculates the cost of deleting a node"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='image')
    parser.add_argument('-o', help='output path')
    parser.add_argument('-l', help='logs directory')
    parser.add_argument('-a', help="artifacts' codebook")
    parser.add_argument('-t', help="trees' codebook")
    parser.add_argument('-d', help='apted custom distance')
    parser.add_argument('-z', help="center's dendogram")

    return parser.parse_args()

def calc_dist_trees(t1, t2, apted_dist):
    #print(t1.tree.name, t2.tree.name)
    return APTED(t1.tree.root, t2.tree.root, apted_dist).compute_edit_distance()

def compute_label(X):
    return np.argwhere(X == np.min(X))[0][0]

def compute_hierarchy(X):
    return np.argwhere(X == np.min(X))[0][0]

def gabi_do_the_math(dendogram, cluster):
    """Computes cluster's neighborhood based on clusters centers' dendogram.
    Author: Gabriela Gomes"""

    def find_index(dendogram, n):
        i = 0
        dendogram_size = dendogram.shape[0]
        while i < dendogram_size and dendogram[i][0] != n and dendogram[i][1] != n: i += 1
        if i == dendogram_size:
            return -1 #no raiz
        return i

    def build_top(dendogram):
        top_info = {}
        alt = {}
        dendogram_size = dendogram.shape[0]
        for d in range(dendogram_size):
            top = find_index(dendogram, dendogram_size + d + 1)
            t = dendogram[top]

            if top in top_info:
                top_info[top].append(d)
            else:
                top_info[top] = [d]

            alt[d] = -1 if top == -1 else top

        return top_info, alt

    def busca1(dendogram, top_info, alt, index, brotherhood):
        dendogram_size = dendogram.shape[0]
        if index == -1:
            return brotherhood

        i, j, dendogram[index][2] = dendogram[index][0], dendogram[index][1], 1
        if i < dendogram_size: brotherhood.append(int(i))
        if j < dendogram_size: brotherhood.append(int(j))
        if index in top_info.keys():
            filhos = top_info[index]
            for f in filhos:
                if dendogram[f][2] == 0:
                    brotherhood = busca1(dendogram, top_info, alt, f, brotherhood)
            if dendogram[alt[index]][2] == 0:
                brotherhood = busca1(dendogram, top_info, alt, alt[index], brotherhood)

        elif dendogram[alt[index]][2] == 0:
            brotherhood = busca1(dendogram, top_info, alt, alt[index], brotherhood)
        
        return brotherhood
    
    dendogram[:,2] = 0

    top_info, alt = build_top(dendogram)

    return busca1(dendogram, top_info, alt, find_index(dendogram, cluster), [])

def main():
    args = init_args()
    o_path = args.o
    log = Log(path=args.l, name='query_image')

    dendogram = np.load(args.z)
    query_img = Image(path=args.i)
    artifacts_clusters = Cluster(load=args.a)
    trees_codebook = np.load(args.t)
    with open(args.d, 'rb') as f:
        apted_dist = pickle.load(f)

    artifacts = query_img.tree.get_tree_masks()[1:]
    
    artifacts_labels = []
    for artifact in artifacts:
        artifacts_labels.append(artifacts_clusters.predict(artifact.feature_vector.reshape(1, -1))[0])

    # TODO: juntar as duas funções abaixo; elas deveriam ser feitas juntas
    query_img.tree.set_labels(artifacts_labels)
    query_img.tree.set_nodes_depth()
    
    M = np.zeros(len(trees_codebook['trees']), np.float_)
    for t1,i in zip(trees_codebook['trees'],range(len(trees_codebook['trees']))):
        M[i] = calc_dist_trees(t1, query_img, apted_dist)
    
    query_img.tree.label = compute_label(M)
    
    label_brotherhood = gabi_do_the_math(dendogram, query_img.tree.label)
    #print(label_brotherhood)
    with open('Master/images/olha_aqui.pickle', 'wb') as f:
        pickle.dump(query_img, f)

    with open(o_path, 'wb') as f:
        pickle.dump({"label":label_brotherhood}, f, protocol=pickle.HIGHEST_PROTOCOL)
	
main()