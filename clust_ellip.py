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

    """This method clusters the samples and return the mean points of each cluster. We will attempt
    to do agglomerative clustering"""

    def cluster(self):
        array = []
        for active_sample in self.points:
            array.append([active_sample.X, active_sample.Y, active_sample.A, active_sample.R])
        centroids = kmeans(obs=array, k_or_guess = 2)
        return None    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""  

    def build_ellipsoids(self):
        return None

    
    """This method is responsible for sampling from the enlarged ellipsoids with certain probability
    The method also checks if any ellipsoids are overlapping and behaves accordingly """
    def sample(self):
        return None



"""Class for ellipsoids"""

class Ellipsoid(object):

    def __init__(self):

        self.covariance_matrix = []
        self.eigenvalues = []
        self.enlarge = []
        self.axislengths = {}




