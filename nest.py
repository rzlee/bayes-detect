import numpy as np
from math import *
from sources import *



class Nested_Sampler(Object):


    """Initialization for the Nested_Sampler"""

    def __init__(self, no_active_samples, max_iter, metropolis=True, cluster=False, plot=False, conv_thresh=0.1):

        """Number of active_samples in the nested sampling loop to start"""
        self.no_active_samples     = no_active_samples

        """Maximum number of iterations after which the loop is terminated"""
        self.maximum_iterations    = max_iter
        
        """Use Metropolis Nested sampling""" 
        self.metropolis            = metropolis

        """Use Clustered Ellipsoidal sampling"""
        self.cluster               = cluster
        self.plot                  = plot

        """Stopping criterion for nested sample. When the evidence changes by less than this , we stop the process"""
        self.convergence_threshold = 0.1

        """ Sampling active points from the prior distribution specified """
        self.active_samples        = get_sources(no_active_samples)
        
        """Information which helps in calculating the uncertainity in evidence"""
        self.information           = None

        """ Variable to hold log_evidence at each iteration"""
        self.log_evidence          = None

        """A Helper variable that we use in information calculation"""
        self.logZ_updated          = None

        """Likelihood constraint for each iteration assigned the value of the point with lowest Likelihood"""
        self.likelihood_constraint = None

        """storing iteration number in case the evidence converges before the maximum number of iterations"""
        self.iteration             = None

        

    
    
    def 
        








