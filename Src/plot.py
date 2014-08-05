import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import *
from matplotlib.patches import Ellipse
import math
from pylab import figure,show
import pickle
from scipy import stats



def show_source(height, width, sources):

    """
    Shows a 2D xy plot of the sources characterized by parameters [X, Y, A, R]

    Parameters
    ----------
    height : int
        Height of the image
    width : int
        width of the image
    sources : array
        Array of source objects to show

    """

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in sources:
        z += i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
    plt.imshow(z)
    plt.show()


def show_samples(height, width, samples):

    """
    Show samples according to their likelihood.

    Parameters
    ----------
    height : int
        Height of the image
    width : int
        width of the image    
    samples : array
        Array of samples to plot according to their likelihood

    """

    min_likelihood = 999999999
    for i in samples:
        if i.logL < min_likelihood:
            min_likelihood = i.logL
    arr = np.zeros((height,width),float)
    for i in samples:
        arr[int(i.Y)][int(i.X)] = i.logL + abs(min_likelihood)
    plt.imshow(arr)
    plt.show()


def plot_histogram(data, bins, title):

    """
    Shows the histogram of a 1D data.

    Parameters
    ----------
    data : array
        data to be shown in histogram format
    bins : int
        number of bins to hold
    title : str
        title of the plot          

    """
    plt.hist(data,bins)
    plt.title(title)
    plt.show()
    return None


def show_scatterplot(X,Y,title, height, width):

    """
    Parameters
    ----------
    X : array
        x coordinates of the data
    Y : array
        y coordinates of the above data. Same shape as above
    title : str
        title of the scatter plot
    height : int
        height limit of the scatter plot                 
    width : int
        width limit of the scatter plot

    """

    plt.scatter(X,Y, marker =".")
    plt.title(title)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()


def plot_ellipse(num_points = 100):

    """
    Plots minimum bounding ellipse around a set of points

    Parameters
    ----------
    num_points : int
        number of points around which ellipse is to be built 

    """
    
    #Getting some test samples from prior to fit an ellipse in X,Y
    Sources = sources.get_sources(num_points)
    X = [[i.X,i.Y] for i in Sources]
    X = np.array(X)
    centroid = np.mean(X,axis=0)
    
    #Transforming the coordinates so that centroid lies at origin
    transformed = X - centroid
    cov_mat = np.cov(m=transformed, rowvar=0)
    inv_cov_mat = np.linalg.inv(cov_mat)
    
    #Calculating a scaling factor for covariance matrix so that all the points lie in the ellipse
    pars = [np.dot(np.dot(transformed[i,:], inv_cov_mat), transformed[i,:]) for i in range(len(X))]
    pars = np.array(pars)
    
    #Corresponds to the point with max value
    scale_factor = np.max(pars)
    cov_mat = cov_mat*scale_factor
    
    #Calculating the width,height and angle of the ellipse from eigen vectors of the scaled covariance matrix
    a, b = np.linalg.eig(cov_mat)
    c = np.dot(b, np.diag(np.sqrt(a)))
    width = np.sqrt(np.sum(c[:,1]**2)) * 2.
    height = np.sqrt(np.sum(c[:,0]**2)) * 2.
    angle = math.atan(c[1,1] / c[0,1]) * 180./math.pi
    
    #Plotting the ellipse
    ellipse = Ellipse(centroid, width, height, angle)
    ellipse.set_facecolor('None')
    plt.figure()
    ax = plt.gca()
    ax.add_patch(ellipse)
    plt.plot(X[:,0], X[:,1], 'ro')
    plt.show()

def write(data, out):

    """ 
    Writes an array to a pickle
    
    Parameters
    ----------
    data : array
        Array to be stored      
    out : str
        The location to be stored at

    """

    f = open(out,'w+b')
    pickle.dump(data, f)
    f.close()


def make_source(src_array,height,width):

    """
    Returns the source image with numpy format

    Parameters
    ----------
    src_array : array
        Array of source objects
    height : int
        height of the image
    width : int
        width of the image

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
    plt.imshow(z)
    plt.title("Source image")
    plt.show()
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

    Returns
    -------
    noised : array
        noised data        

    """
    height = len(data)
    width = len(data[0])
    my_noise=stats.distributions.norm.rvs(mean,sd,size=(height, width))
    noised = data + my_noise
    plt.imshow(noised)
    plt.title("Source image with additive gaussian noise of rms "+str(sd)+" units")
    plt.show()
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
    plt.imshow(z)
    plt.title("Source image")
    plt.show()
    return z



if __name__ == '__main__':
        srces =  [[43.71, 22.91, 10.54, 3.34],
                  [101.62, 40.60, 1.37, 3.40],
                  [92.63, 110.56, 1.81, 3.66],
                  [183.60, 85.90, 1.23, 5.06],
                  [34.12, 162.54, 1.95, 6.02],
                  [153.87, 169.18, 1.06, 6.61],
                  [155.54, 32.14, 1.46, 4.05],
                  [130.56, 183.48, 1.63, 4.11]]
        make_source(srces, 200, 200)              







       