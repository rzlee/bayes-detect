import numpy as np

from sampler import Sampler
from source import Source

class Uniform_Sampler(Sampler):

    """
    An Implementation of uniform sampling to randomly pick a sample satisfying the likelihood
    constraint in the current nested sampling phase.

    Attributes
    ----------
    LC : float
        likelihood constraint for the point
    number : int
        likelihood calculations until now

    """

    def __init__(self, data_map, params):

        """
        Initializes the uniform sampler

        Parameters
        ----------
        likelihood_constraint: float
            name says it all
        no : int
            Number of likelihood evaluations until this point

        """
        Sampler.__init__(self, data_map, params)


    def sample(self):

        """
        Method to pick the sample satisfying the likelihood constraint using uniform sampling

        Returns
        -------
        new : object
            The evolved sample
        number : int
            Number of likelihood calculations after sampling

        """

        new = Source()

        x_l, x_u = self.getPrior_X()
        y_l, y_u = self.getPrior_Y()
        r_l, r_u = self.getPrior_R()
        a_l, a_u = self.getPrior_A()

        new.X = np.random.uniform(x_l,x_u)
        new.Y = np.random.uniform(y_l,y_u)
        new.A = np.random.uniform(a_l,a_u)
        new.R = np.random.uniform(r_l,r_u)
        new.logL = self.log_likelihood(new)

        return new
