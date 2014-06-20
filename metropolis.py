""" Metropolis hastings sampler to be used in Multimodal nested sampling technique
for source detection in astronomical images. """

import numpy as np
from math import *


class Metropolis_sampler(Object):

	def __init__(self, to_evolve, likelihood_constraint):

		self.source = to_evolve
		self.LC     = likelihood_constraint


	def proposal_distribution(self):


    
    def sample(self):

		
	



