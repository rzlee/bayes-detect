import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from math import *
import random
from nest import *
from plot import *

"""Reading the Image data from fits file"""
fitsFile = "simulated_images/ufig_20_g_sub_500_sub_small.fits"

hdulist   = fits.open(fitsFile)
data_map   = (hdulist[0].data)
height, width = len(data_map), len(data_map[0])
no_pixels = width*height

"""Converting the data_map into a vector for likelihood calculations"""
data_map = data_map.flatten()

"""Bounds for the prior distribution of Amplitude """
amplitude_upper = 1.4*np.max(data_map)
amplitude_lower = np.mean(data_map) + 2*np.std(data_map)

"""Bounds for the prior distribution of position """
x_upper = 400
y_upper = 100

"""Bounds for the prior distribution of Spatial extent """
R_upper = 3.0
R_lower = 2.5

PI = np.pi

"""Incorporating RMS noise into the model"""    
noise = 2.0 
K = (no_pixels/2)*(np.log(2*PI) + 2*np.log(noise))

"""Useful in likelihood evaluation for calculating the simulated object as the function of indices"""
x_forcalc = np.arange(0, 400)
y_forcalc = np.arange(0, 100)
xx, yy = np.meshgrid(x_forcalc, y_forcalc, sparse=True)

"""Number of objects used in nested_sampling"""
n = 40

"""Number of Iterations for nested_sampling method """
max_iterations = 1500



"""Object Information"""
class Source:
    def __init__(self):
        self.X = None
        self.Y = None
        self.A = None
        self.R = None
        self.logL = None
        self.logWt = None

def log_likelihood(Source):
    simulated_map = Source.A*np.exp(-1*((xx-Source.X)**2+(yy-Source.Y)**2)/(2*(Source.R**2)))
    diff_map = data_map - simulated_map.flatten()
    return -0.5*np.dot(diff_map, np.transpose((1/(noise**2))*diff_map)) - K    
    
    
def proposed_model(x, y, X, Y, A, R):
    return A*np.exp(((x-X)**2 + (y-Y)**2)/(2*(R**2)))
    
"""Sampling the object from prior distribution"""
def sample_from_prior():
    src = Source()
    src.X = random.uniform(0.0, x_upper)
    src.Y = random.uniform(0.0, y_upper) 
    src.A = random.uniform(amplitude_lower, amplitude_upper)
    src.R = random.uniform(R_lower, R_upper)
    src.logL = log_likelihood(src)
    return src

"""Method to replace the lowest likelihood object in each iteration of nested_sampling"""
def explore(src, logLstar):
    ret = Source()
    ret.__dict__ = src.__dict__.copy()
    Try = Source();
    step = 40 # for (0,400)
    hit = 0
    miss = 0
    for i in range(20):
        # Trial object
        Try.X = ret.X + step * (2.*random.uniform(0,1) - 1.);  # |move| < step
        Try.Y = ret.Y + (step/4) * (2.*random.uniform(0,1) - 1.);  # |move| < step
        if(Try.X >= width or Try.X < 0): Try.X = ret.X;
        if(Try.Y >=height or Try.Y < 0): Try.Y = ret.Y
        Try.A = random.uniform(amplitude_lower, amplitude_lower)
        Try.R = random.uniform(R_lower, R_upper)
        Try.logL = log_likelihood(Try);

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret.__dict__ = Try.__dict__.copy()
            hit+=1
        else:
            miss+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( hit > miss ):   step *= exp(1.0 / hit);
        if( hit < miss ):   step /= exp(1.0 / miss);

    return ret

if __name__ == '__main__':
    a = nested_sampling(n, max_iterations, sample_from_prior, explore)
    show_samples(100, 400, a["samples"])         