import sys, os, pickle, math
import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

from sklearn import metrics
from sklearn.metrics import pairwise_distances

'''
	sklearn source code with some changes in relation to the distance metric

	http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#example-cluster-plot-mean-shift-py


@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
'''

def get_mean_node(points, M):

	dist = np.zeros((points.shape[0],))

	for i in range(points.shape[0]):
		D = M[points[i],points]
		dist[i] = np.sum(D)

	pos=np.argmin(dist)

	return points[pos]

def compute_kneighbors(X, k, Y=None):

	dist = X
	if Y is not None:
		dist = X[:,Y]

	S = np.argsort(dist, axis=1)

	idx = S[:,range(k)]

	M = np.zeros((idx.shape))
	i=0
	for sa in S:
		M[i,:] = dist[i,sa[:k]]
		i+=1

	return M, idx

def estimate_bandwidth(X):
	"""Estimate the bandwidth to use with MeanShift algorithm

		Parameters
		----------
		X : array [n_samples, n_features]
		Input points.

		quantile : float, default 0.3
		should be between [0, 1]
		0.5 means that the median of all pairwise distances is used.

		n_samples : int
		The number of samples to use. If None, all samples are used.

		random_state : int or RandomState
		Pseudo number generator state used for random sampling.

		Returns
		-------
		bandwidth : float
		The bandwidth parameter.
	"""

	quantile=0.3
	random_state=0
	random_state = check_random_state(random_state)

	knn = int(X.shape[0] * quantile)
	#nbrs = NearestNeighbors(n_neighbors=int(X.shape[0] * quantile))
	#nbrs.fit(X)

	d, _ = compute_kneighbors(X, knn)
	bandwidth = np.mean(np.max(d, axis=1))

	return bandwidth

def compute_radius_neighbors(X, M, radius, points=None):
	"""Finds the neighbors within a given radius of a point or points.

	Returns indices of the neighbors of each point.

	Parameters
	----------
	X : array-like, last dimension same as that of fit data
	The new point or points

	radius : float
	Limiting distance of neighbors to return.
	(default is the value passed to the constructor).

	return_distance : boolean, optional. Defaults to True.
	If False, distances will not be returned

	Returns
	-------
	dist : array
	Array representing the euclidean distances to each point,
	only present if return_distance=True.

	ind : array
	Indices of the nearest points in the population matrix.

	Examples
	--------
	In the following example, we construct a NeighborsClassifier
	class from an array representing our data set and ask who's
	the closest point to [1,1,1]

	>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
	>>> from sklearn.neighbors import NearestNeighbors
	>>> neigh = NearestNeighbors(radius=1.6)
	>>> neigh.fit(samples) # doctest: +ELLIPSIS
	NearestNeighbors(algorithm='auto', leaf_size=30, ...)
	>>> print(neigh.radius_neighbors([1., 1., 1.])) # doctest: +ELLIPSIS
	(array([[ 1.5, 0.5]]...), array([[1, 2]]...)

	The first array returned contains the distances to all points which
	are closer than 1.6, while the second array returned contains their
	indices. In general, multiple points can be queried at the same time.

	Notes
	-----
	Because the number of neighbors of each point is not necessarily
	equal, the results for multiple query points cannot be fit in a
	standard data array.
	For efficiency, `radius_neighbors` returns arrays of objects, where
	each object is a 1D array of indices or distances.
	"""

	#X = atleast2d_or_csr(X)
	if points is None:
		points = range(M.shape[1])

	dist = M[X, points]
	neigh_ind = np.where(dist < radius)[0]

	# if there are the same number of neighbors for each point,
	# we can do a normal array. Otherwise, we return an object
	# array with elements that are numpy arrays
	try:
		neigh_ind = np.asarray(neigh_ind, dtype=int)
		dtype_F = float
	except ValueError:
		neigh_ind = np.asarray(neigh_ind, dtype='object')
		dtype_F = object

	return neigh_ind

def mean_shift(X, bandwidth=None):
	"""
	Perform MeanShift Clustering of data using a flat kernel

	Seed using a binning technique for scalability.

	Parameters
	----------

	X : matrix of distance (source X source)
	Input data.

	bandwidth : float, optional
	Kernel bandwidth.
	If bandwidth is not defined, it is set using
	a heuristic given by the median of all pairwise distances.

	seeds : array [n_seeds, n_features]
	Point used as initial kernel locations.

	bin_seeding : boolean
	If true, initial kernel locations are not locations of all
	points, but rather the location of the discretized version of
	points, where points are binned onto a grid whose coarseness
	corresponds to the bandwidth. Setting this option to True will speed
	up the algorithm because fewer seeds will be initialized.
	default value: False
	Ignored if seeds argument is not None.

	min_bin_freq : int, optional
	To speed up the algorithm, accept only those bins with at least
	min_bin_freq points as seeds. If not defined, set to 1.

	Returns
	-------

	cluster_centers : array [n_clusters, n_features]
	Coordinates of cluster centers.

	labels : array [n_samples]
	Cluster labels for each point.

	Notes
	-----
	See examples/cluster/plot_meanshift.py for an example.

	"""

	min_bin_freq=1
	max_iterations=300
	bandwidth = bandwidth if bandwidth != None else estimate_bandwidth(X)
	seeds = range(X.shape[0]) #get_bin_seeds(X, bandwidth, min_bin_freq)

	n_samples = X.shape[0]
	stop_thresh = 1e-3 * bandwidth # when mean has converged
	center_intensity_dict = {}

	#nbrs = NearestNeighbors(radius=bandwidth).fit(X)

	# For each seed, climb gradient until convergence or max_iterations
	for my_mean in seeds:
		completed_iterations = 0

		while True:
			# Find mean of points within bandwidth
			#i_nbrs =
			points_within = compute_radius_neighbors(my_mean, X, bandwidth)


			#points_within = X[i_nbrs]
			if points_within.size == 0:
				break # Depending on seeding strategy this condition may occur
			my_old_mean = my_mean # save the old mean
			my_mean = get_mean_node(points_within, X)
			# If converged or at max_iterations, addS the cluster
			if (X[my_mean, my_old_mean] < stop_thresh or
					completed_iterations == max_iterations):
				center_intensity_dict[my_mean] = points_within.size
				break
			completed_iterations += 1

    # POST PROCESSING: remove near duplicate points
    # If the distance between two kernels is less than the bandwidth,
    # then we have to remove one because it is a duplicate. Remove the
    # one with fewer points.

	sorted_by_intensity = sorted(center_intensity_dict.items(),
                                 key=lambda tup: tup[1], reverse=True)

	sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])

	unique = np.ones(len(sorted_centers), dtype=np.bool)

    #nbrs = NearestNeighbors(radius=bandwidth).fit(sorted_centers)
	for i, center in enumerate(sorted_centers):
		if unique[i]:
			neighbor_idxs = compute_radius_neighbors(center, X, bandwidth, sorted_centers)
			unique[neighbor_idxs] = 0
			unique[i] = 1 # leave the current point as unique
	cluster_centers = sorted_centers[unique]

	
    # ASSIGN LABELS: a point belongs to the cluster that it is closest to
	#nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)

    # Not necessary
	labels = np.zeros(n_samples, dtype=np.int)
	distances, idxs = compute_kneighbors(X, 1, cluster_centers)

	labels = idxs.flatten() #cluster_all == True
	#labels.fill(-1) #cluster_all == False

	bool_selector = distances.flatten() <= bandwidth
	labels[bool_selector] = idxs.flatten()[bool_selector]

	'''
	#AVALIACAO
	D = X[~np.eye(X.shape[0],dtype=bool)].reshape(X.shape[0],-1)
	print("silhouette:", metrics.silhouette_score(D, labels, metric='mahalanobis'))
	print("calinski_harabaz:", metrics.calinski_harabaz_score(D, labels))
	'''

	return np.array(cluster_centers) , np.array(labels)

def save_codebook(centers, labels, trees_file, output):
	
	img_trees = []
	with open(trees_file, 'rb') as f:
		while True:
			try:
				img_trees.append(pickle.load(f))
			except EOFError:
				break
	
	D = {}
	T = []
	for c in centers:
		D[labels[c]] = c
		img_trees[c].tree.label = labels[c] #já dá a label aqui msm
		T.append(img_trees[c])


	np.savez(output + '_codebook.npz', map=D, trees=T) #dict mapping C -> L
 
	return D, T

def save_centers(centers, labels, output):
	np.save(output + '_centers.npy', centers)
	np.save(output + '_vq-train.npy', labels) #o index da lista de centroids corresponde a label associada

def main():

	matrix_file = sys.argv[1]
	output = sys.argv[2]
	
	trees_file = sys.argv[4]

	M = np.load(matrix_file)

	bandwidth = estimate_bandwidth(M) if sys.argv[3] == 'None' else float(sys.argv[3])

	print("Bandwidth: {}".format(bandwidth))
	# with open('exec/results-batch.txt', 'a') as f:
	# 	f.write("- BWValue = {}\n".format(bandwidth))
	# with open('exec/results-batch2.txt', 'a') as f:
	# 	f.write("- asdBWValue = {}\n".format(bandwidth))

	C, L = mean_shift(M, bandwidth)
	with open('exec/results-batch.txt', 'a') as f:
		f.write('C_size = {}\n'.format(C.size))
	with open('exec/results-batch2.txt', 'a') as f:
		f.write('C_size = {}\n'.format(C.size))
	
	save_centers(C, L, output)
	D, T = save_codebook(C, L, trees_file, output)

	print(C, C.size)
	print(L, L.size)
	print(D, len(D))


	# TODO:Análise do cluster 0; todo código pode ser removido
	# cluster0_indexes = np.where(L == 0)[0]
	# cluster0_size = cluster0_indexes.shape[0]
	# cluster0_matrix = np.zeros((cluster0_size,cluster0_size), dtype=np.float_)

	# ai = 0
	# for x,i in zip(M,range(M.shape[0])):
	# 	aj = 0
	# 	if (i not in cluster0_indexes):
	# 		continue
	# 	for y,j in zip(x,range(M.shape[0])):
	# 		if(j not in cluster0_indexes):
	# 			continue
	# 		cluster0_matrix[ai,aj] = M[i,j]
	# 		aj += 1
	# 	ai += 1
	
	# print(cluster0_matrix[1])

	# cluster0_bandwidth = estimate_bandwidth(cluster0_matrix)
	# print("cluster0_bandwidth: {}".format(cluster0_bandwidth))
	# cluster0_C, cluster0_L = mean_shift(cluster0_matrix, cluster0_bandwidth)
	# print(cluster0_C, cluster0_C.size)
	# print(cluster0_L, cluster0_L.size)
main()
