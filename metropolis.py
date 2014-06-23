""" Metropolis hastings sampler to be used in Multimodal nested sampling technique
for source detection in astronomical images.

Metropolis hastings principle helps in generating a point satisfying likelihood constraint
as follows:
1.Start with a point a
2.Generate a point using a symmetric proposal distribution with a as center and a stepsize
as the dispersion. In choosing a symmetric distribution we gain the advantage of transitional
probabilities being equal.
3.Accept the point with probability min(1., priorRatio*LikelihoodRatio)
4.Since our prior is a uniform distribution in all dimensions we just accept the point if it has a
greater likelihood
5.We need to pass through the likelihood contours, Inorder restrict or increase the prior volume
we choose an accept-reject ratio. We do this by running it in a loop of 20 steps.
6.When the accept-reject ratio is good the stepsize is decreased to reduce the prior volume.
If it is bad then stepsize is increased to facilitate for looking in other possible regions  

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

    def __init__(self, to_evolve, likelihood_constraint, no):

        self.source = to_evolve
        self.LC     = likelihood_constraint
        self.start  = 10.0
        self.step   = 10.0
        self.number = no
        
    """We use a symmetric normal distribution for generating new point"""
    def generate_point(self, obj, step):
        x_l, x_u = getPrior_X()
        y_l, y_u = getPrior_Y()
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A()

        stepX    = step
        stepY    = (step/self.start)*(y_u-y_l)
        stepA    = (step/self.start)*(a_u-a_l)
        stepR    = (step/self.start)*(r_u-r_l)

        new      = sample_source()
        new.X    = np.random.normal(obj.X, (stepX))        
        new.Y    = np.random.normal(obj.Y, (stepY))
        new.A    = np.random.normal(obj.A, (stepA))
        new.R    = np.random.normal(obj.R, (stepR))

        if(new.X > x_u or new.X < x_l): new.X = obj.X;
        if(new.Y > y_u or new.Y < y_l): new.Y = obj.Y;
        if(new.A > a_u or new.A < a_l): new.A = obj.A;
        if(new.R > r_u or new.R < r_l): new.R = obj.R;        
        
        new.logL = log_likelihood(new)

        return new  

    """Sampling from the prior subject to constraints according to Metropolis method 
    proposed by Sivia et al discussed in Multinest paper by feroz and Hobson"""

    def sample(self):
        metro = self.source
        self.number+=1
        next  = None
        hit   = 0
        miss  = 0

        for i in range(20):
            next = self.generate_point(metro,abs(self.step))
            self.number+=2
            if(next.logL > metro.logL):
                metro.__dict__ = next.__dict__.copy()
                hit+=1
            else:
                miss+=1

        if( hit > miss ):   self.step *= exp(1.0 / hit);
        if( hit < miss ):   self.step /= exp(1.0 / miss);        
                                
        return metro, self.number