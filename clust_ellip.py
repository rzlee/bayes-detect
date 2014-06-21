"""Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
and Hobson(2008). This method is closer in spirit to the recursive technique advocated in Shaw
et al. 

References :
===========

Multinest paper by Feroz and Hobson 2008.
Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)
A Nested Sampling Algorithm for Cosmological Model Selection(2006) Pia Mukherjee , David Parkinson , and Andrew R. Liddle 

"""

import numpy as np
from math import *
from sources import *
import random
from scipy.cluster.vq import vq, kmeans2


class Clustered_Sampler(object):

    """Initialize using the information of current active samples and the object to evolve"""
    def __init__(self, to_evolve, active_samples, likelihood_constraint):

        self.source = to_evolve
        self.points = active_samples
        self.LC     = likelihood_constraint
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.ellipsoid_set = None
        self.activepoint_set = self.build_set()

    def build_set(self):
        array = []
        for active_sample in self.points:
            array.append([active_sample.X, active_sample.Y, active_sample.A, active_sample.R])
        return array            
        

    """This method clusters the samples and return the mean points of each cluster. We will attempt
    to do agglomerative clustering as an improvement in coming days""" 

    def cluster(self, activepoint_set, number_of_clusters):
        centroids, label = kmeans(obs=activepoint_set, k_or_guess = number_of_clusters)
        return centroids, label, activepoint_set    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""  

    def optimal_ellipsoids(self):
        volume_condition = False
        overlap_condition = False
        clust_points = None
        ellipsoids = []
        #clust_cent = self.cluster(self.tempClust,2)
        while(volume_condition==False or overlap_condition ==False):
            #volume = 
            clust_cent, point_labels, pointset = self.cluster(self.tempClust,2)#Cluster and find centroids
            clust_points = np.empty(len(clust_cent))
            ellipsoids = np.empty(len(clust_cent))
            #Find points for each cluster
            for i in range(len(clust_cent)):
                clust_points[i] = [pointset(x) for x in range(len(pointset)) if(label(x)==i) ]
            #Build ellipsoids
            for i in range(len(clust_cent)):
                ellipsoids[i] = self.build_ellipsoid(centroid=clust_cent[i], points=clust_points[i])

            #Check the combined volume of ellipsoids
            #Check if any ellipsoids are overlapping
            #If both conditions are satisfied return the ellipsoids
            #Else continue the process again until they satisfy

        return None

    
    def build_ellipsoid(self, centroid, points):
        return Ellipsoid(centroid = centroid, points = points, enlargement_factor=1.5)

    
    def check_overlapping(self):
        return None


    def sample_from_ellipsoid(self):
        return None
        
        
        

    


    """This method is responsible for sampling from the enlarged ellipsoids with certain probability
    The method also checks if any ellipsoids are overlapping and behaves accordingly """
    def sample(self):
        return None



"""Class for ellipsoids"""

class Ellipsoid(object):

    def __init__(self, centroid = None, points, enlargement_factor):

        self.centroid = centroid
        self.clpoints = points
        self.enlargement_factor =enlargement_factor
        self.covariance_matrix = self.build_cov(centroid, points)
        self.eigenvalues = self.get_eigenvalues(self.covariance_matrix)
        self.semi_axes = self.find_semiaxes(eigenvalues=self.eigenvalues)
        self.volume = self.find_volume()

    
    def build_cov(self, center, clpoints):
        return np.cov(m=clpoints, rowvar=0)


    def get_eigenvalues(self,covariance_matrix):
        return np.linalg.eigvals(self.covariance_matrix)

    
    def find_semiaxes(self, eigenvalues):
        axes = np.reciprocal(np.sqrt(eigenvalues))
        self.enlarge(semiaxes=axes)
        return axes

    def find_volume(self):
        semi_axes = self.semi_axes
        volume = 0.5*(np.pi**2)*(np.product(semi_axes)) 
        return volume    


    def enlarge(self,semiaxes):
        for i in semiaxes:
            i = i*self.enlargement_factor
        



class SingleEllipsoidal_Sampler(object):

    def __init__(self, to_evolve, activepoint_set, likelihood_constraint):

        self.points = activepoint_set
        self.source = to_evolve
        self.LC = likelihood_constraint
        self.ellipsoid = self.build_ellipsoid(points = activepoint_set, enlargement_factor = )

    def build_ellipsoid(self, points, enlargement_factor):
        return Ellipsoid(points = points, enlargement_factor = enlargement_factor)



