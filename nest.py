""" Calculating the evidence and the posterior samples using the Multimodal Nested sampling method.
Multimodal nested sampling by FEROZ and HOBSON is a method built on nested sampling algorithm
by JOHN SKILLING et al for sampling through posterior with multiple modes. The method uses a loop 
where we continously replace the point in active samples which has lower likelihood with a point that
has greater likelihood. The discarded point is a posterior inference and used in calculation of evidence.
The method has the following steps:
1)Sample points from prior
2)Loop under stopping criterion
  1) Find out the point with lowest likelihood-L(i)
  2) Assign Prior mass for this iteration-X(i) 
  3) Set the values of weights using the trapezoidal rule-W(i)
  4) Increment the evidence by lowest_likelihood- L(i)*W(i) and store the point as posterior inference.
  5) Check stopping criterion. If converged Increment the evidence by using the active_samples too
  6) If not converged, sample a point from prior with likelihood greater than L(i)
3) After reaching maximum number of iterations Increment the evidence by using the active_samples too
4) Output the Evidence, the posterior samples and other important information 
The Difficult part in this method is sampling a new point from the prior satisfying the likelihood constraint.
According to the Multinest paper by Feroz et al. there are some methods that could help restrict the prior 
volume from which to draw a point ensuring transition between a lower iso-likelihood contour to a higher
iso-likelihood contour. The methods are :
1) Clustered ellipsoidal sampling - Builds ellipsoids around midpoints of clustered active_samples and draws
samples from these ellipsoids with a certain probability assigned to each ellipsoid.
2) Metropolis nested sampling - Uses a proposal distribution generally a symmetric gaussian distribution with a 
dispersion value which changes and drives the process to higher likelihood regions as we sample.
We are going to try both the metropolis nested sampling and clustered ellipsoidal sampling for this approach.
The Following is a class implementation of Nested_Sampler."""


import numpy as np
from math import *
from sources import *



class Nested_Sampler(Object):


    """Initialization for the Nested_Sampler"""

    def __init__(self, no_active_samples, max_iter, sample = "metropolis", plot=False, conv_thresh=0.1):

        """Number of active_samples in the nested sampling loop to start"""
        self.no_active_samples     = no_active_samples

        """Maximum number of iterations after which the loop is terminated"""
        self.maximum_iterations    = max_iter
        
        """ The sampling method used to draw a new sample satisfying the likelihood constraint"""
        self.sample                = sample

        """Plot posterior samples while running the loop to check how the method is working"""
        self.plot                  = plot

        """Stopping criterion for nested sample. The same value used in the Multinest paper """
        self.convergence_threshold = 0.1

        """ Sampling active points from the prior distribution specified """
        self.active_samples        = get_sources(no_active_samples)
        
        """ Variable to hold evidence at each iteration"""
        self.evidence              = None

        """ Posterior samples for evidence and plotting """
        self.posterior_inferences  = []

        """ Prior volume which is used to calculate the weight of the point at each iteration"""
        self.prior_volume          = None

    
    """ Method that runs the main nested sampling loop"""
    
    def fit(self):  
