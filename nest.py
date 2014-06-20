import numpy as np
from math import *
from sources import *



class Nested_Sampler(Object):


    """Initialization for the Nested_Sampler"""

    def __init__(self, no_active_samples, max_iter, metropolis=True, cluster=False, plot=False, conv_thresh=0.1):

        self.no_active_samples     = no_active_samples
        self.maximum_iterations    = max_iter
        self.metropolis            = metropolis
        self.cluster               = cluster
        self.plot                  = plot
        self.convergence_threshold = conv_thresh
        self.active_samples        = get_sources(no_active_samples)
        self.log_evidence          = None
        self.logZ_updated          = None
        self.likelihood_constraint = None
        self.iteration             = None
        self.information           = None

    
    
        








