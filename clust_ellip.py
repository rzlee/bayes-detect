"""Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
and Hobson(2008). This method is closer in spirit to the recursive technique advocated in Shaw
et al. 

References :
===========

Multinest paper by Feroz and Hobson 2008.
Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)

"""

import numpy as np
from math import *
from sources import *
import random
from scipy.cluster.vq import vq, kmeans


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

    # FIX ME: Change it to recursive clustering with the stopping criterion as the following
    # 1) The combined volume of the ellipsoids is less than that of the previous state
    # 2) The ellipsoids are properly seperated by a user defined value 

    def cluster(self, number_of_clusters):
        centroids = kmeans(obs=self.activepoint_set, k_or_guess = number_of_clusters)
        return centroids    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""  

    def optimal_ellipsoids(self):
        #Loop
            #Cluster and find centroids
            #Build ellipsoids
            #Check the combined volume of ellipsoids
            #Check if any ellipsoids are overlapping
            #If both conditions are satisfied return the ellipsoids
            #Else continue the process again until they satisfy

        return None

    
    def build_ellipsoids(self):
        return None


    def find_volume(self):
        return None


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

    def __init__(self):

        self.covariance_matrix = []
        
    def enlarge(self):
        return None
        