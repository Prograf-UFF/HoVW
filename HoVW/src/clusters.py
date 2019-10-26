import os, argparse, time, re, pickle, gc
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from klepto.archives import file_archive
from utils import Log


class Cluster(MiniBatchKMeans, MeanShift):
    """A general class which includes others SKLearn.Cluster
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster

    Now, we implment both KMeans and MeanShift.

    Parameters
    ----------
    cnt: Shape's contour.
    outline: Shape's outline.

    Attributes
    ----------
    All attributes available on parental classes.
    centroids_distance: array, shape = [codebook size, codebook size]
        Dissimilarity matrix (distances) between clusters centroids.
    """

    def __init__(self, **kwargs):
        self.centroids_distance = None
        if 'load' in kwargs:
            self._load(kwargs['load'])
        else:
            try:
                method = kwargs['method']
                del kwargs['method']
            except KeyError as ke:
                print(kr)
                raise

            if method == 'KMeans' or method == 'MiniBatchKMeans':
                MiniBatchKMeans.__init__(self,**kwargs)
            elif method == 'MeanShift':
                MeanShift.__init__(self,**kwargs)
            else:
                e = "No method '{}' avaiable. Please use KMeans or MeanShift".format(method)
                log = Log(path='.', name='cluster_class')
                log.write(error=e, data=self)
                raise ValueError(e)

    def load(self, path):
        """Load the model object from a serialized file.
        
        Parameters
        ----------
        path: string
            Path where the file is in.
        """

        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
    
    def save(self, path, name=None):
        """Save the model object in a serialized file.
        
        Parameters
        ----------
        path: string
            Path where the file should be save.
        name: string, default = None
            Name of the file.
        """
        #TODO: centroids_distance nao deve ser aqui
        
        self.centroids_distance = self._centroids_distance_matrix()

        print("Saving", name, "in", path)
        arch = file_archive('{}.clr'.format(os.path.join(path, name)))
        for d in self.__dict__:
            arch[d] = self.__dict__[d]
        arch.dump()
        
        with open(os.path.join(path, name) + '2.clr', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Exporting", name + "-cdm", "in", path)
        np.save(os.path.join(path, name + "-cdm") + '.npy', self.centroids_distance)
    
    def _centroids_distance_matrix(self):
        """Generate the dissimilarity matrix of the clusters centroids.
            In a Bag-of-Words apporach the codebook is iqual the 
            number of clusters in the model.

        Returns
        -------
        Array, shape = [codebook size, codebook size]
            Dissimilarity matrix (distances) between clusters centroids.
        """

        return distance_matrix(self.cluster_centers_, self.cluster_centers_)

    _load = load