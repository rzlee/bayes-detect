import numpy as np
from scipy import stats

def make_source(src_array, height, width):

    """
    Returns the source image with numpy format

    Parameters
    ----------
    src_array : array
        Array of source objects in format [X,Y,A,R]
    height : int
        height of the image
    width : int
        width of the image
    display : bool
        whether or not to display the plot

    Returns
    -------
    z : array
        Source image in numpy format
    """

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in src_array:
        z+= i[2]*np.exp(-1*((xx-i[0])**2+(yy-i[1])**2)/(2*(i[3]**2)))
    return z


def add_gaussian_noise(mean, sd, data):

    """
    Adds indpendent gaussian noise of given rms units

    Parameters
    ----------
    mean : float
        Mean of the gaussian . We mostly use zero
    sd : float
        Standard deviation (i.e RMS) of the noise
    data : array
        data to which the noise is added
    display : bool
        whether or not to display the plot

    Returns
    -------
    noised : array
        noised data
    """


    height = len(data)
    width = len(data[0])
    my_noise=stats.distributions.norm.rvs(mean,sd,size=(height, width))
    noised = data + my_noise
    return noised

def make_random_source(limits, width, height, number_of_sources):

    """
    Makes an Image with randomly distributed sources

    Parameters
    ----------
    limits : 2Darray
        prior limits
    width : int
        width of the image
    length : int
        length of the image
    number_of_sources : int
        Number of sources
    display : bool
        whether or not to display the plot

    Returns
    -------
    z : array
        Source Image
    """

    x_l, x_u = limits[0]
    y_l, y_u = limits[1]
    a_l, a_u = limits[2]
    r_l, r_u = limits[3]

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in range(number_of_sources):
        z+= np.random.uniform(a_l,a_u)*np.exp(-1*((xx-np.random.uniform(x_l,x_u))**2+(yy-np.random.uniform(y_l,y_u))**2)/(2*(np.random.uniform(r_l,r_u)**2)))
    return z
