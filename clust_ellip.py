"""Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
and Hobson(2008). This method is closer in spirit to the recursive technique advocated in Shaw
et al. 

References :
===========

Multinest paper by Feroz and Hobson 2008.
Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)
A Nested Sampling Algorithm for Cosmological Model Selection(2006) Pia Mukherjee , David Parkinson , and Andrew R. Liddle
http://www.sjsu.edu/faculty/watkins/ellipsoid.htm 

"""

import numpy as np
from math import *
from sources import *
import random
from scipy.cluster.vq import vq, kmeans2
import copy
from sklearn.cluster import AffinityPropagation


class Clustered_Sampler(object):

    """Initialize using the information of current active samples and the object to evolve"""
    def __init__(self, active_samples, enlargement):

        self.points = copy.deepcopy(active_samples)
        self.enlargement = enlargement
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.ellipsoid_set = None
        self.activepoint_set = self.build_set()
        self.total_vol = None

    def build_set(self):
        array = []
        for active_sample in self.points:
            array.append([float(active_sample.X), float(active_sample.Y)])
        return np.array(array)            
        

    # FIX ME : This method clusters the samples and return the mean points of each cluster. We will attempt
    #to do agglomerative clustering as an improvement in coming days""" 

    def cluster(self, activepoint_set):
        af = AffinityPropagation().fit(activepoint_set)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        print str(labels)
        number_of_clusters= len(cluster_centers_indices)
        print str(number_of_clusters)
        return number_of_clusters, labels, activepoint_set    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""

    def optimal_ellipsoids(self):
        number_of_clusters, point_labels, pointset = self.cluster(self.activepoint_set)#Cluster and find centroids
        clust_points = np.empty(number_of_clusters,dtype=object)
        ellipsoids = np.empty(number_of_clusters,dtype=object)
        for i in range(number_of_clusters):
            clust_points[i] = [pointset[x] for x in range(len(pointset)) if(point_labels[x]==i)]
        for i in range(number_of_clusters):
            if len(clust_points[i]) > 1:
                try:
                    ellipsoids[i] = Ellipsoid(points=clust_points[i],
                              enlargement_factor = 1.0)#enlargement*np.sqrt(len(self.activepoint_set)/len(clust_points[i]))
                except np.linalg.linalg.LinAlgError:
                    ellipsoids[i] = None
                    print str(i)
            else:
                ellipsoids[i] = None
                
        return ellipsoids

    
    """This method is responsible for sampling from the enlarged ellipsoids with certain probability
    The method also checks if any ellipsoids are overlapping and behaves accordingly """
    def sample(self):
        return None


"""Class for ellipsoids"""

class Ellipsoid(object):

    def __init__(self, points, enlargement_factor):

        self.clpoints = points
        self.centroid = np.mean(points,axis=0)
        self.enlargement_factor = enlargement_factor
        self.covariance_matrix = self.build_cov(self.centroid, self.clpoints)

    
    def build_cov(self, center, clpoints):
        points = np.array(clpoints)
        transformed = points - center
        cov_mat = np.cov(m=transformed, rowvar=0)
        inv_cov_mat = np.linalg.inv(cov_mat)
        pars = [np.dot(np.dot(transformed[i,:], inv_cov_mat), transformed[i,:]) for i in range(len(transformed))]
        pars = np.array(pars)
        scale_factor = np.max(pars)
        cov_mat = cov_mat*scale_factor*self.enlargement_factor
        return cov_mat

    def sample(self):
        return None
        


if __name__ == '__main__':
    sources = get_sources(300)
    clustellip = Clustered_Sampler(active_samples = sources, enlargement= 1.0)
    X = [i.X for i in sources]
    Y = [i.Y for i in sources]
    ellipsoids = clustellip.optimal_ellipsoids()
    plt.figure()
    ax = plt.gca()
    for i in range(len(ellipsoids)):
        if ellipsoids[i]!=None:
            a, b = np.linalg.eig(ellipsoids[i].covariance_matrix)
            c = np.dot(b, np.diag(np.sqrt(a)))
            width = np.sqrt(np.sum(c[:,1]**2)) * 2.
            height = np.sqrt(np.sum(c[:,0]**2)) * 2.
            angle = math.atan(c[1,1] / c[0,1]) * 180./math.pi
            ellipse = Ellipse(ellipsoids[i].centroid, width, height, angle)
            ellipse.set_facecolor('None')
            ax.add_patch(ellipse)
    plt.plot(X, Y, 'ro')
    plt.show()
     


