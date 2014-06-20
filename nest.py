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
  3) Set the values of weights using the trapezoidal rule-W(i) or calculating the information
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
The Following is a class implementation of Nested_Sampler.

References:
===========
Multinest paper by Feroz and Hobson et al(2008)
Nested Sampling by John Skilling et al
http://www.inference.phy.cam.ac.uk/bayesys/
"""


import numpy as np
from math import *
from sources import *
from metropolis import *
import random



class Nested_Sampler(object):


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

        """ Prior mass which is used to calculate the weight of the point at each iteration"""
        self.log_width             = None

        """ Information for calculating the uncertainity of calculating the evidence """
        self.Information           = None

    
    """ Method that runs the main nested sampling loop"""
    
    def fit(self):

        """ Initializing evidence and prior mass """
        self.log_evidence = -1e300
        self.log_width = log(1.0 - exp(-1.0 / n))
        self.Information = 0.0

        for iteration in range(self.maximum_iterations):
            if iteration%100==0: print str(iteration);# temporary verbose
            smallest = 0 
            
            """Finding the object with smallest likelihood"""
            for i in range(self.no_active_samples):
                if self.active_samples[i].logL < self.active_samples[smallest].logL:
                    smallest = i
            
            """Assigning local evidence to the smallest sample"""
            self.active_samples[smallest].logWt = self.log_width + self.active_samples[smallest].logL;
            
            """Calculating the updated evidence"""
            temp_evidence = np.logaddexp(self.log_evidence, self.active_samples[smallest].logWt)
            
            """Calculating the information which will be helpful in calculating the uncertainity"""
            self.Information = exp(self.active_samples[smallest].logWt - temp_evidence) * self.active_samples[smallest].logL + \
            exp(self.log_evidence - temp_evidence) * (self.Information + self.log_evidence) - temp_evidence;
            
            """assigning the updated evidence in this iteration"""
            if(temp_evidence-self.log_evidence <= self.convergence_threshold): break;
            self.log_evidence = temp_evidence
            
            """storing posterior points"""
            self.posterior_inferences.append(self.active_samples[smallest])
            
            """New likelihood constraint""" 
            likelihood_constraint = self.active_samples[smallest].logL

            """Picking an object to evolve for next iteration"""
            while True:
                 k = int(self.no_active_samples*random.uniform(0,1))
                 if k!=smallest:
                    break
            self.active_samples[smallest] = self.active_samples[k]

            """Drawing a new sample satisfying the likelihood constraint"""  
            
            if self.sample == "metropolis":
                """Obtain new sample using Metropolis principle"""
                self.active_samples[smallest] = self.metropolis_sampling(obj = self.active_samples[smallest], LC = likelihood_constraint)

            if self.sample == "clustered_ellipsoidal":
                """Obtain new sample using Clustered ellipsoidal sampling"""
                self.active_samples[smallest] = self.clustered_sampling(obj = self.active_samples[smallest], LC = likelihood_constraint)
            
            """Shrink width"""  
            self.log_width -= 1.0 / self.no_active_samples;

        # FIX ME: Incorporate the active samples into evidence calculation and information after the loop """

        return {"samples":self.posterior_inferences, 
            "logZ":self.log_evidence,
            "Information":self.Information
            }


    """ Method for drawing a new sample using the metropolis hastings principle """  

    def metropolis_sampling(self, obj, LC):
        "Instantiating the metropolis sampler object"
        Metro = Metropolis_sampler(to_evolve = obj, likelihood_constraint = LC)
        evolved = Metro.sample()
        return evolved


    """ Method for drawing a new sample using clustered ellipsoidal sampling"""
    
    def clustered_sampling(self, obj, LC):
        return None

    











            


                



                    










