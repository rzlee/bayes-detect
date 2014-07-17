import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import *
#from sources import *
from matplotlib.patches import Ellipse
import math
from pylab import figure,show
import pickle
from scipy import stats 


"""Method to show the sources at the end of the nested sampling method"""
def show_source(height, width, sources):
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in sources:
        z += i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
    plt.imshow(z)
    plt.show()

"""Method to show the samples at the end of the nested sampling method"""
def show_samples(height, width, samples):
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
    plt.hist(data,bins)
    plt.title(title)
    plt.show()
    return None

# FIX ME : Figure out a way to embed information and
# show on the plot to scale and also shouls be immune to interactive transformations
def show_scatterplot(X,Y,title, height, width):
    plt.scatter(X,Y, marker =".")
    plt.title(title)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()


"""Method to show the sources at the end of the nested sampling method"""
def show_source():
    height =100
    width = 400
    i = Source()
    i.X = 200
    i.Y = 50
    i.A = 100
    i.R = 3
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    z = i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
    plt.imshow(z)
    plt.show()


def plot_ellipse():
    
    """ Getting some test samples from prior to fit an ellipse in X,Y"""
    sources = get_sources(100)
    X = [[i.X,i.Y] for i in sources]
    X = np.array(X)
    centroid = np.mean(X,axis=0)
    
    """Transforming the coordinates so that centroid lies at origin"""
    transformed = X - centroid
    cov_mat = np.cov(m=transformed, rowvar=0)
    inv_cov_mat = np.linalg.inv(cov_mat)
    
    """Calculating a scaling factor for covariance matrix so that all the points lie in the ellipse"""
    pars = [np.dot(np.dot(transformed[i,:], inv_cov_mat), transformed[i,:]) for i in range(len(X))]
    pars = np.array(pars)
    
    """ Corresponds to the point with max value"""
    scale_factor = np.max(pars)
    cov_mat = cov_mat*scale_factor
    
    """Calculating the width,height and angle of the ellipse from eigen vectors of the scaled covariance matrix"""
    a, b = np.linalg.eig(cov_mat)
    c = np.dot(b, np.diag(np.sqrt(a)))
    width = np.sqrt(np.sum(c[:,1]**2)) * 2.
    height = np.sqrt(np.sum(c[:,0]**2)) * 2.
    angle = math.atan(c[1,1] / c[0,1]) * 180./math.pi
    
    """Plotting the ellipse"""
    ellipse = Ellipse(centroid, width, height, angle)
    ellipse.set_facecolor('None')
    plt.figure()
    ax = plt.gca()
    ax.add_patch(ellipse)
    plt.plot(X[:,0], X[:,1], 'ro')
    plt.show()

def write(data, out):
    f = open(out,'w+b')
    pickle.dump(data, f)
    f.close()


def make_source(src_array,height,width):
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
    height = len(data)
    width = len(data[0])
    my_noise=stats.distributions.norm.rvs(mean,sd,size=(height, width))
    noised = data + my_noise
    plt.imshow(noised)
    plt.title("Source image with additive gaussian noise of rms "+str(sd)+" units")
    plt.show()
    return noised

    

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







       