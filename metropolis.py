""" Metropolis hastings sampler to be used in Multimodal nested sampling technique
for source detection in astronomical images.

References:
===========
Multinest paper by Feroz and Hobson et al(2008) 
Data Analysis: A Bayesian Tutorial by Sivia et al(2006)
http://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
http://www.inference.phy.cam.ac.uk/bayesys/
 """

import numpy as np
from math import *
from sources import *
import random


class Metropolis_sampler(object):

    """Initializing metropolis sampler"""

    def __init__(self, to_evolve, likelihood_constraint):

        self.source = to_evolve
        self.LC     = likelihood_constraint


    """Sampling from the prior subject to constraints according to Metropolis method 
    proposed by Sivia et al """

    def sample(self):

        metro = Source()
        metro.__dict__ = self.source.__dict__.copy()
        step = 40 # for X (0, 400) will be scaled down to 10 for Y
        hit = 0
        miss = 0
        Trial = Source()
        x_l, x_u = getPrior_X()
        y_l, y_u = getPrior_Y()
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A() #trial object

        for i in range(20):

            """ Evolving the chain from lower likelihood contour to higher likelihood contour"""
            Trial.X = metro.X + step*(2.0*random.uniform(0,1)-1.0)
            Trial.Y = metro.Y + ((step/x_u)*(y_u-y_l)) *(2.0*random.uniform(0,1)-1.0)
            Trial.A = metro.A + ((step/x_u)*(a_u-a_l)) *(2.0*random.uniform(0,1)-1.0)
            Trial.R = metro.R + ((step/x_u)*(r_u-r_l)) *(2.0*random.uniform(0,1)-1.0)

            """Ensure source attributes stay with in prior"""
            if(Trial.X > x_u or Trial.X < x_l): Trial.X = metro.X;
            if(Trial.Y > y_u or Trial.Y < y_l): Trial.Y = metro.Y;
            if(Trial.A > a_u or Trial.A < a_l): Trial.A = metro.A;
            if(Trial.R > r_u or Trial.R < r_l): Trial.R = metro.R;
            
            """Calculating the likelihood of the evolved point"""
            Trial.logL = log_likelihood(Trial)

            """Accept if and only if within hard likelihood constraint"""
            if Trial.logL > self.LC:
                metro.__dict__ = Trial.__dict__.copy()
                hit+=1
            else:
                miss+=1

            """Refine step-size to let acceptance ratio converge around 50 percent as discussed in multinest paper"""
            if( hit > miss ):   step *= exp(1.0 / hit);
            if( hit < miss ):   step /= exp(1.0 / miss);

        return metro     







        
    



