import numpy as np
import copy

class Ellipsoid(object):

    """
    An Implementation of minimum bounding ellipsoids for use in Ellipsoidal methods.

    Attributes
    ----------
    clpoints : array
       array of points
    centroid : array
       centroid of the ellipsoid
    enlargement_factor : float
       factor of enlargement
    covariance_matrix : array
       covariance matrix of the points of the ellipsoid
    inv_cov_mat : array
       inverse covariance matrix of the points of the ellipsoid
    volume : float
       Volume of the ellipsoid

    References
    ----------
    .. [1] Multinest paper by Feroz and Hobson 2008.
    .. [2] Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)

    """

    def __init__(self, params, points, enlargement_factor):

        """
        Initializes the ellipsoid object

        Parameters
        ----------
        points : array
            The point set for the minimum bounding ellipsoid
        enlargement_factor : float
            Enlargement factor for better sampling

        """

        self.clpoints = points
        self.centroid = np.mean(points,axis=0)
        self.enlargement_factor = enlargement_factor
        self.covariance_matrix = self.build_cov(self.centroid, self.clpoints)
        self.inv_cov_mat = np.linalg.inv(self.covariance_matrix)
        self.params = params

    def getPrior_X(self):
        return (0.0, self.params['width'])
    def getPrior_Y(self):
        return (0.0, self.params['height'])
    def getPrior_R(self):
        return (self.params['r_lower'], self.params['r_upper'])
    def getPrior_A(self):
        return (self.params['amp_lower'], self.params['amp_upper'])

    def build_cov(self, center, clpoints):

        """
        Builds the scaled covariance matrix such that the ellipsoid encloses all the points.

        Parameters
        ----------
        center : array
            The centroid of the point cluster
        clpoints : array
            The point set of interest

        Returns
        -------
        cov_mat : array
            Scaled covariance matrix

        """

        points = np.array(clpoints)
        transformed = points - center
        cov_mat = np.cov(m=transformed, rowvar=0)
        inv_cov_mat = np.linalg.inv(cov_mat)
        pars = [np.dot(np.dot(transformed[i,:], inv_cov_mat), transformed[i,:]) for i in range(len(transformed))]
        pars = np.array(pars)
        scale_factor = np.max(pars)
        cov_mat = cov_mat*scale_factor*self.enlargement_factor
        return cov_mat


    def sample(self, n_points):

        """
        Method to sample points inside the ellipsoid

        Parameters
        ----------
        n_points : int
            Number of points to sample

        Returns
        -------
        points : array
            The array of sampled points

        """


        dim = 2
        points = np.empty((n_points, dim), dtype = float)
        values, vects = np.linalg.eig(self.covariance_matrix)
        x_l, x_u = self.getPrior_X()
        y_l, y_u = self.getPrior_Y()
        r_l, r_u = self.getPrior_R()
        a_l, a_u = self.getPrior_A()
        scaled = np.dot(vects, np.diag(np.sqrt(np.absolute(values))))
        bord = 1
        new = None
        for i in range(n_points):
            while bord==1:
                bord = 0
                randpt = np.random.randn(dim)
                point  = randpt* np.random.rand()**(1./dim) / np.sqrt(np.sum(randpt**2))
                new =  np.dot(scaled, point) + self.centroid

                if(new[0] > x_u or new[0] < x_l): bord = 1;
                if(new[1] > y_u or new[1] < y_l): bord = 1;

            bord = 1
            points[i, :] = copy.deepcopy(new)
        return points


    def find_volume(self):

        """
        The method to find the volume of ellipsoid

        Returns
        -------
        volume : float
            volume of the ellipsoid under consideration

        """

        volume = (np.pi**2)*(np.sqrt(np.linalg.det(self.covariance_matrix)))/2.
        return volume
    
