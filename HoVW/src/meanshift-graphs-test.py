import sys, os, pickle
import numpy as np
from image import Image
from apted import APTED, Config
from apted.helpers import Tree
from apted import APTED, PerEditOperationConfig

class my_distance(Config):
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
		if node1.label == None or node2.label == None:
			return 0 #atenção aqui, se for raiz e tentar renomear a raiz dá zero. precisa ser melhor pensado
		return self.centroids_matrix[node1.label][node2.label]

	def children(self, node):
		return node.childs

def compute_kneighbors(X):
	labels = []
	for x in X:
		l = np.argwhere(x == np.min(x))[0][0]
		labels.append(l)

	return labels

def compute_distance_matrix(test_imgs, codebook_trees, cmdist):
	M = np.zeros((len(test_imgs), len(codebook_trees)), np.float_)
	for i in range(len(test_imgs)):
		print(i)
		t1 = test_imgs[i]
		M[i] = ([APTED(t1.tree.root, t2.tree.root, my_distance(cmdist)).compute_edit_distance() for t2 in codebook_trees])

	return M

def assign_labels(test_imgs, labels, output):
	print("Assigning labels")
	with open(output, 'wb') as handle:
		for t, l in zip(test_imgs, labels):
			t.tree.label = l
			print(t.tree.name, l)
			pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():

	test_file = sys.argv[1]
	codebook_file = sys.argv[2]
	cdist_file = sys.argv[3]
	cluster_centers_file = sys.argv[4]

	test_imgs = []
	with open(test_file, 'rb') as f:
		while True:
			try:
				test_imgs.append(pickle.load(f))
			except EOFError:
				break
	codebook = np.load(codebook_file)
	codebook_map = codebook['map']
	codebook_trees = codebook['trees']
	a = np.load(cdist_file)
	cluster_centers = np.load(cluster_centers_file)
	
	M = compute_distance_matrix(test_imgs, codebook_trees, a)

	labels = compute_kneighbors(M)

	assign_labels(test_imgs, labels, 'Master/data/img_trees_labeled_full-test.pickle')
	
if __name__ == '__main__':
	main()

