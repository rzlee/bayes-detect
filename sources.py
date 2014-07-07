"""Bayesian Source detection and characterization in astronomical images

References:
===========
Multinest paper by Feroz and Hobson et al(2008) 
Data Analysis: A Bayesian Tutorial by Sivia et al(2006)
http://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
http://www.inference.phy.cam.ac.uk/bayesys/
Hobson, Machlachlan, Bayesian object detection in astronomical images(2003)
"""


import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from math import *
import random
from nest import *
from plot import *
import time
import pickle

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
amplitude_lower = np.mean(data_map)

"""Bounds for the prior distribution of position """
x_upper = 400.0
y_upper = 100.0

"""Bounds for the prior distribution of Spatial extent """
R_upper = 3.0
R_lower = 2.0

PI = np.pi

"""Incorporating RMS noise into the model"""    
noise = 2.0#stats.mode(data_map) 
K = (no_pixels/2)*(np.log(2*PI) + 4*np.log(abs(noise)))

"""Useful in likelihood evaluation for calculating the simulated object as the function of indices"""
x_forcalc = np.arange(0, 400)
y_forcalc = np.arange(0, 100)
xx, yy = np.meshgrid(x_forcalc, y_forcalc, sparse=True)

"""Number of objects used in nested_sampling"""
n = 1200

"""Number of Iterations for nested_sampling method """
max_iterations = 8000


"""Object Information 
   X : x coordinate of the center of the object
   Y : y coordinate of the center of the object
   A : Amplitude of the object
   R : Spatial extent of the object
   logL : Log likelihood of the object
   logWt: Local evidence of the object """

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
def sample_source():
    src = Source()
    src.X = random.uniform(0.0, x_upper)
    src.Y = random.uniform(0.0, y_upper) 
    src.A = random.uniform(amplitude_lower, amplitude_upper)
    src.R = random.uniform(R_lower, R_upper)
    src.logL = log_likelihood(src)
    return src

"""Method which helps the nested sampler to generate a number of active samples"""
def get_sources(no_active_points):
    src_array = []
    for i in range(no_active_points):
        src_array.append(sample_source())
    return src_array

"""This method returns the prior bounds of amplitude of a source"""
def getPrior_A():
    return amplitude_lower, amplitude_upper;

"""This method returns the prior bounds of amplitude of a source"""
def getPrior_R():
    return R_lower, R_upper;

""" This method returns the prior bounds of X value"""
def getPrior_X():
    return 0.0, width;

""" This method returns the prior bounds of Y value"""
def getPrior_Y():
    return 0.0, height;

def write(data, out):
    f = open(out,'w+b')
    pickle.dump(data, f)
    f.close()

def read(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data



if __name__ == '__main__':
        startTime = time.time()
        nest = Nested_Sampler(no_active_samples = n, max_iter = max_iterations)
        out  = nest.fit()
        elapsedTime = time.time() - startTime
        print "elapsed time: "+str(elapsedTime) 
        print "log evidence: "+str(out["logZ"])
        print "number of iterations: "+str(out["iterations"])
        print "likelihood calculations: "+str(out["likelihood_calculations"])
        dispersion = 10
        data = np.array(out["samples"])
        write(data,"sub_"+str(max_iterations)+"_"+str(n)+"_"+str(dispersion))
        outX = [i.X for i in out["samples"]]
        outY = [100-i.Y for i in out["samples"]]   
        plot_histogram(data = outX, bins = 400)
        plot_histogram(data = outY, bins = 100)
        show_scatterplot(outX,outY)
        outsrcX = [i.X for i in out["src"]]
        outsrcY = [100-i.Y for i in out["src"]]
        plot_histogram(data = outsrcX, bins = 400)
        plot_histogram(data = outsrcY, bins = 100)
        show_scatterplot(outsrcX,outsrcY)
           
         


