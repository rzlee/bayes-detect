import numpy as np
import copy

from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans2

from ellipsoid import Ellipsoid
from sampler import Sampler
from source import Source

class Clustered_Sampler(Sampler):

    """
    Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
    an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
    and Hobson(2008).

    Attributes
    ----------
    points : array
        copy of active samples in the current nested sampling phase
    LC : float
        likelihood_constraint
    enlargement : float
        Enlargement factor used in constructing ellipsoids
    clustered_point_set : array
        The points belonging to different clusters
    number_of_clusters : int
        Number of clusters
    activepoint_set : float
        array of activepoint_set for clustering
    ellipsoid_set : array
        array of optimal ellipsoids constructed around the active point set.
    total_vol : float
        Total volume enclosed by the optimal ellipsoids
    number : float
        Number of likelihood evaluations until now

    References
    ----------
    .. [1] Multinest paper by Feroz and Hobson 2008.
    .. [2] Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)
    .. [3] A Nested Sampling Algorithm for Cosmological Model Selection(2006) Pia Mukherjee , David Parkinson , and Andrew R. Liddle
    .. [4] http://www.sjsu.edu/faculty/watkins/ellipsoid.htm

    """


    def __init__(self, data_map, params, active_samples, likelihood_constraint, enlargement, no):

        """
        Initializes the clustered ellipsoidal sampler.

        Parameters
        ----------
        data_map : array
            Data map we want to sample on
        params : dict
            Dictionary of various parameters
        active_samples : array
            The array containing the active samples for this clustered sampling phase
        likelihood_constraint : float
            Name says it all
        enlargement : float
            Enlargement factor for ellipsoid
        no : int
            Number of likelihood calculations until the current sampling phase

        """

        Sampler.__init__(self, data_map, params)

        self.eps = params['eps']
        self.minPts = params['minPts']
        self.params = params
        #self.points = copy.deepcopy(active_samples)
        self.LC = likelihood_constraint
        self.enlargement = 1.5
        self.clustered_point_set = None
        self.number_of_clusters = None
        activepoint_set = self.build_set(active_samples)
        self.ellipsoid_set = self.optimal_ellipsoids(activepoint_set)
        self.total_vol = None
        self.number = no



    def build_set(self, points):

        """
        Builds set of the active samples in array format for clustering.

        Returns
        -------
        array : array
            Array containing the active samples. Each element [X,Y] represents an object

        """
        array = [[float(active_sample.X), float(active_sample.Y)] for active_sample in points]
        return np.array(array)


    def cluster(self, activepoint_set):

        """
        Clusters an array of samples using DBSCAN

        Parameters
        ----------
        activepoint_set : array
            Set of active points to be clustered

        Returns
        -------
        number_of_clusters : int
            name says it all
        labels : array
            Cluster labels assigned by dbscan for each sample
        activepoint_set : array
            The point set

        """

        db = DBSCAN(eps=self.eps, min_samples=self.minPts).fit(activepoint_set)
        labels = db.labels_
        labelset = set(labels) #its O(1) to check membership in set vs O(n) in list
        number_of_clusters = len(labelset) - (1 if -1 in labelset else 0)
        return number_of_clusters, labels, activepoint_set


    def optimal_ellipsoids(self, activepoint_set):

        """

        Method for calculating ellipsoids around the individual clusters.

        Returns
        -------
        ellipsoids : array
            An array of ellipsoids to sample from.

        """

        self.number_of_clusters, point_labels, pointset = self.cluster(activepoint_set)#Cluster and find centroids
        clust_points = np.empty(self.number_of_clusters,dtype=object)
        ellipsoids = np.empty(self.number_of_clusters,dtype=object)
        for i in range(self.number_of_clusters):
            clust_points[i] = np.array(pointset[np.where(point_labels==i)])
        invalid = []
        for i in range(self.number_of_clusters):
            if len(clust_points[i]) > 1:
                try:
                    ellipsoids[i] = Ellipsoid(self.params, points=clust_points[i],
                                              enlargement_factor = 2.0)
                except np.linalg.linalg.LinAlgError:
                    ellipsoids[i] = None
                    invalid.append(i)
            else:
                ellipsoids[i] = None
                invalid.append(i)
        ellipsoids = np.delete(ellipsoids, invalid)
        return ellipsoids


    def recursive_bounding_ellipsoids(self, data, ellipsoid=None):

        """
        Implementation of finding minimum bounding ellipsoids recursively. (work in progress)

        Parameters
        ----------
        data : array
            The active samples around which ellipsoids are to be built.
        ellipsoid : object
            Starting ellipsiod. None by default

        Returns
        -------
        ellipsoids : array
           Array of ellipsoids satisfying all the conditions

        """

        ellipsoids = []
        if ellipsoid is None:
            ellipsoid = Ellipsoid(self,params, points = data, enlargement_factor=1.0)
        centroids, labels = kmeans2(data, 2, iter=10)
        clustered_data = [None, None]
        clustered_data[0] = [data[i] for i in range(len(data)) if labels[i]==0]
        clustered_data[1] = [data[i] for i in range(len(data)) if labels[i]==1]
        vol = [0.0,0.0]
        clustered_ellipsoids = np.empty(2,object)
        for i in [0, 1]:
            if(len(clustered_data[i]) <= 1):
                clustered_ellipsoids[i] = Ellipsoid(self.params, clustered_data[i],1.0)
                vol[i]= clustered_ellipsoids[i].volume
        do = True

        if(vol[0] + vol[1] < ellipsoid.volume ):
            for i in [0, 1]:
                if(vol[i]>0.0):
                    ellipsoids.extend(self.recursive_bounding_ellipsoids(
                                      np.array(clustered_data[i]), clustered_ellipsoids[i]))
        else:
            ellipsoids.append(ellipsoid)
        print len(ellipsoids)
        return ellipsoids

    def run_clustering(self, active_samples):
        """
        Run clustering on this set of active samples

        Parameters
        _________
        active_samples : array
            The active samples to cluster on
        """
        activepoint_set = self.build_set(active_samples)
        self.ellipsoid_set = self.optimal_ellipsoids(activepoint_set)

    def sample(self):

        """
        Sampling from the built ellipsoids.

        Returns
        -------
        clust : object
            The sampled point satisfying the likelihood constraint
        number : int
            The number of likelihood calculations until this

        """

        trial = Source()
        clust = Source()
        z = int(np.random.uniform(0, len(self.ellipsoid_set)))
        points = None
        try:
            points = self.ellipsoid_set[z].sample(n_points=50)
        except IndexError:
            raise Exception("Adjust clustering parameters or number of active samples")
        max_likelihood = self.LC
        count = 0
        r_l, r_u = self.getPrior_R()
        a_l, a_u = self.getPrior_A()
        while count<50:
            trial.X = points[count][0]
            trial.Y = points[count][1]
            trial.A = np.random.uniform(a_l,a_u)
            trial.R = np.random.uniform(r_l,r_u)
            trial.logL = self.log_likelihood(trial)
            self.number+=1

            if(trial.logL > max_likelihood):
                clust.__dict__ = trial.__dict__.copy()
                max_likelihood = trial.logL
                break

            count += 1

        return clust, self.number
