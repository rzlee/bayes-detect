import numpy as np

from math import exp
from sampler import Sampler
from source import Source

class Metropolis_Sampler(Sampler):

    """
    An Implementation of Metropolis sampling to pick a sample satisfying the likelihood
    constraint in the current nested sampling phase.

    Attributes
    ----------
    source : object
       object to evolve
    LC : float
        likelihood constraint for the point
    step : float
        dispersion of the gaussian proposal distribution
    number : int
        likelihood calculations until now

    """

    def __init__(self, data_map, params, to_evolve, likelihood_constraint, no):

        """
        Initializes the Metropolis sampler

        Parameters
        ----------
        to_evolve : object
            The sample to evolve
        likelihood_constraint: float
            name says it all
        no : int
            Number of likelihood evaluations until this point

        """
        Sampler.__init__(self, data_map, params)

        self.source = to_evolve
        self.LC     = likelihood_constraint
        self.step   = params['dispersion']
        self.number = no


    def sample(self):

        """
        Method to pick the sample satisfying the likelihood constraint using metropolis sampling

        Returns
        -------
        metro : object
            The evolved sample
        number : int
            Number of likelihood calculations until now

        """

        metro = Source()
        metro.__dict__ = self.source.__dict__.copy()
        start = Source()
        start.__dict__ = self.source.__dict__.copy()
        new   = Source()
        self.number+=1
        count = 0
        hit = 0
        miss = 0

        x_l, x_u = self.getPrior_X()
        y_l, y_u = self.getPrior_Y()
        r_l, r_u = self.getPrior_R()
        a_l, a_u = self.getPrior_A()

        stepnormalize = self.step/x_u

        stepX    = self.step
        stepY    = stepnormalize*(y_u-y_l)
        stepA    = stepnormalize*(a_u - a_l)
        stepR    = stepnormalize*(r_u-r_l)

        bord = 1

        while(count<20):

            while bord==1:
                bord = 0
                new.X    = metro.X + stepX * (2.*np.random.uniform(0, 1) - 1.);
                new.Y    = metro.Y + stepY * (2.*np.random.uniform(0, 1) - 1.);
                new.A    = metro.A + stepA * (2.*np.random.uniform(0, 1) - 1.);
                new.R    = metro.R + stepR * (2.*np.random.uniform(0, 1) - 1.);



                if(new.X > x_u or new.X < x_l): bord = 1;
                if(new.Y > y_u or new.Y < y_l): bord = 1;
                if(new.A > a_u or new.A < a_l): bord = 1;
                if(new.R > r_u or new.R < r_l): bord = 1;

            new.logL = self.log_likelihood(new)
            self.number+=1

            if(new.logL > self.LC):
                metro.__dict__ = new.__dict__.copy()
                hit+=1
            else:
                miss+=1

            if( hit > miss ):   self.step *= exp(1.0 / hit);
            if( hit < miss ):   self.step /= exp(1.0 / miss);

            stepnormalize = self.step/x_u

            stepX    = self.step
            stepY    = stepnormalize*(y_u-y_l)
            stepA    = stepnormalize*(a_u - a_l)
            stepR    = stepnormalize*(r_u-r_l)


            count+=1
            bord=1

        return metro, self.number

