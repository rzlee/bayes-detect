import random
import numpy as np
import abc

from source import Source

class Sampler(object):
    """
    Base class with implementations of common functions that all samplers need to function.
    Note that there is an abstract function sample() that every subclass is required to implement.
    """

    def __init__(self, data_map, params):
        self.data_map = data_map
        self.x_upper = params['x_upper']
        self.y_upper = params['y_upper']
        self.amplitude_lower = params['amp_lower']
        self.amplitude_upper = params['amp_upper']
        self.R_lower = params['r_lower']
        self.R_upper = params['r_upper']
        self.K = params['k']
        self.noise = params['noise']
        self.height = params['height']
        self.width = params['width']

        x_forcalc = np.arange(0, self.width)
        y_forcalc = np.arange(0, self.height)
        (self.xx, self.yy) = np.meshgrid(x_forcalc, y_forcalc, sparse=True)
    

    def log_likelihood(self, src_obj):

        """
        Returns the log likelihood of the source object.

        Parameters
        ----------
        src_obj : object
            A source object.
        ------
            TypeError : When we pass an object with any of X, Y, A, R attributes as None type

        """

        simulated_map = src_obj.A*np.exp(-1*((self.xx-src_obj.X)**2+(self.yy-src_obj.Y)**2)/(2*(src_obj.R**2)))
        diff_map = self.data_map - simulated_map.flatten()
        return -0.5*np.dot(diff_map, np.transpose((1/(self.noise**2))*diff_map)) - self.K

    def sample_source(self):

        """
        Sampling the object from prior distribution.

        Returns
        -------
        src : object
            The source object with X,Y,A,R sampled from their prior distribution and log likelihood calculated.


        """

        src = Source()
        src.X = random.uniform(0.0, self.x_upper)
        src.Y = random.uniform(0.0, self.y_upper)
        src.A = random.uniform(self.amplitude_lower, self.amplitude_upper)
        src.R = random.uniform(self.R_lower, self.R_upper)
        src.logL = self.log_likelihood(src)
        return src
        

    def sample_source(self):

        """
        Sampling the object from prior distribution.

        Returns
        -------
        src : object
            The source object with X,Y,A,R sampled from their prior distribution and log likelihood calculated.


        """

        src = Source()
        src.X = random.uniform(0.0, self.x_upper)
        src.Y = random.uniform(0.0, self.y_upper)
        src.A = random.uniform(self.amplitude_lower, self.amplitude_upper)
        src.R = random.uniform(self.R_lower, self.R_upper)
        src.logL = self.log_likelihood(src)
        return src


    def get_sources(self, num_active_points):

        """
        Returns an array of source objects sampled from their prior distribution.

        Parameters
        ----------
        num_active_points : int
            The number of source objects to be returned

        Returns
        -------
        src_array : array
            An array of objects with size equal to num_active_points.

        """

        return [self.sample_source() for i in range(num_active_points)]


    def getPrior_A(self):

        """
        Returns
        -------
        bounds : tuple
            a tuple of the amplitude bounds.

        """
        return (self.amplitude_lower, self.amplitude_upper)


    def getPrior_R(self):

        """
        Returns
        -------
        bounds : tuple
            a tuple of the R bounds.

        """
        return (self.R_lower, self.R_upper)


    def getPrior_X(self):

        """
        Returns
        -------
        bounds : tuple
            a tuple of the X bounds.

        """
        return (0.0, self.width)


    def getPrior_Y(self):

        """
        Returns
        -------
        bounds : tuple
            a tuple of the Y bounds.

        """
        return (0.0, self.height)

    @abc.abstractmethod
    def sample(self):
        """
        Draw a sample.
        We implemented it as an abstract method
        because we want every subclass to implement this.
        This should greatly simplify the code as we can easily switch
        samplers within the nested sampler's loop.
        """
        return
