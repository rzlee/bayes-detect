# Bayesian Source detection and characterization
# Authors : Krishna Chaitanya Chavati, Richard Lee
# Email  : chaithukrishnazz2@gmail.com, rlee46@illinois.edu

import sys
import numpy as np
from astropy.io import fits
from astropy.io import ascii
from math import *
import plot
import time
import os
import platform
import pickle

from nested import Nested_Sampler

Config = {}

config_found = 1

data_map = None

try:
    filepath = os.path.dirname(os.path.realpath(__file__))
    if platform.system() == "Windows":
        filepath = filepath+"\config.cfg"
    else:
        filepath = filepath+"/config.cfg"
    #workaround for windows and unix having opposite slashes
    config_file = open(filepath)
except IndexError:
    print "Can't find the config file"
    config_found = 0

if config_found:
    params = filter(lambda x: (x[0] != '#') and (x[0] != '\n'), config_file.readlines())
    #only read the lines with params
    params = [line.rstrip('\n').split('=') for line in params]
    #strip extraneous characters
    Config = dict(params)
    #store everything into (dict) Config

    #note: config has the absolute file path
    image_path = (Config['IMAGE_PATH'])

    if image_path.endswith(".fits"):
        hdulist    = fits.open(image_path)
        data_map   = (hdulist[0].data)

    else:
        with open(image_path, 'r') as f:
            data_map = pickle.load(f)

else:
    image_path = os.path.join(os.path.dirname(__file__), os.pardir)
    image_path = image_path + "/assets/simulated_images/multinest_toy_noised"

    if image_path.endswith(".fits"):
        hdulist   = fits.open(image_path)
        data_map   = (hdulist[0].data)

    else:
        with open(image_path, 'r') as f:
            data_map = pickle.load(f)


height, width = len(data_map), len(data_map[0])
no_pixels = width*height

#Converting the data_map into a vector for likelihood calculations
data_map = data_map.flatten()


#Useful in likelihood evaluation for calculating the simulated object as the function of indices
x_forcalc = np.arange(0, width)
y_forcalc = np.arange(0, height)
xx, yy = np.meshgrid(x_forcalc, y_forcalc, sparse=True)

def run_source_detect(samples = None, iterations = None, sample_method = None, prior= None,noise_rms = None, disp = None,mode = "Manual" ):

    """
    The main method for Bayesian source detection. Runs and generates plots and histograms for posterior samples and
    active samples. Outputs the source positions, posterior inferences and other details for reference and analysis.

    Parameters
    ----------
    samples : int
        Number of active samples for nested sampling
    iterations : int
        Maximum number of iterations for nested sampling
    sample_method : str
        The method for sampling. example : "metropolis", "uniform", "clustered_ellipsoidal"
    prior : array
        Prior distribution for all parameters of the sources
    noise_rms : float
        The RMS noise for modelling the likelihood function
    disp : float
        dispersion used in metropolis method
    mode : str
        Running mode

        * "Manual" : Can be ran manually form the Command line . In this case all the above params are obtained
                   from the config file.
        * "ipython" : Can be ran independently from ipython console. In this case we have to provide all the parameters

    Notes
    -----
    Generates the following plots successively.

        * X-histogram of the posterior samples
        * Y-histogram of the posterior samples
        * Scatter plot of the posterior samples in 2D i.e (X,Y)
        * X-histogram of the active samples
        * Y-histogram of the active samples
        * Scatter plot of the active samples in 2D i.e (X,Y)

    Outputs the posterior samples and the following information to ASCII file

        * Time elapsed
        * Log evidence
        * Number of iterations
        * Number of likelihood evaluations

    """

    startTime = time.time()
    params = dict()

    if mode == "ipython":
        params['dispersion'] = disp
        params['amp_upper'] = prior[2][1] 
        params['amp_lower'] = prior[2][0] 
        params['x_upper'] = prior[0][1] 
        params['y_upper'] = prior[1][1] 
        params['r_upper'] = prior[3][1] 
        params['r_lower'] = prior[3][0] 
        params['noise'] = noise_rms
        params['k'] = (no_pixels/2)*(np.log(2*np.pi) + 4*np.log(abs(noise_rms)))
        params['n'] = samples
        params['max_iter'] = iterations
        params['type'] = sample_method

        #need to check os to get path right
        filepath = os.path.join(os.path.dirname(__file__), os.pardir)
        if platform.system() == "Windows":
            filepath = filepath+"\samples.dat"
        else:
            filepath = filepath+"/samples.dat"

        params['output_loc'] = filepath 
        params['stop'] = 0
        params['eps'] = 10
        params['minPts'] = 10
        params['stop_by_evidence'] = 1

    else:
        params['dispersion'] = float(Config['DISPERSION'])
        params['amp_upper'] = float(Config['A_PRIOR_UPPER'])
        params['amp_lower'] = float(Config['A_PRIOR_LOWER'])
        params['x_upper'] = float(Config['X_PRIOR_UPPER'])
        params['y_upper'] = float(Config['Y_PRIOR_UPPER'])
        params['r_upper'] = float(Config['R_PRIOR_UPPER'])
        params['r_lower'] = float(Config['R_PRIOR_LOWER'])
        params['noise'] = float(Config['NOISE'])
        params['k'] = (no_pixels/2)*(np.log(2*np.pi) + 4*np.log(abs(params['noise'])))
        params['n'] = int(Config['ACTIVE_POINTS'])
        params['max_iter'] = int(Config['MAX_ITER'])
        params['type'] = str(Config['SAMPLER'])
        params['output_loc'] = str(Config['OUTPUT_DATA_PATH'])
        params['stop'] = int(Config['STOP_BY_EVIDENCE'])
        params['eps'] = float(Config['EPS'])
        params['minPts'] = float(Config['MINPTS'])
        params['stop_by_evidence'] = int(Config['STOP_BY_EVIDENCE']) 

    params['width'] = width
    params['height'] = height
    
    nested = Nested_Sampler(data_map, params, sampler = params['type'])
    out  = nested.fit()

    elapsedTime = time.time() - startTime
    print "elapsed time: " + str(elapsedTime)
    print "log evidence: " + str(out["logZ"])
    print "number of iterations: " + str(out["iterations"])
    print "likelihood calculations: " + str(out["likelihood_calculations"])

    data = np.array(out["samples"])

    X = np.array([i.X for i in data])
    Y = np.array([i.Y for i in data])
    A = np.array([i.A for i in data])
    R = np.array([i.R for i in data])
    logL = np.array([i.logL for i in data])

    ascii.write([X, Y, A, R, logL], output_loc, names=['X', 'Y', 'A', 'R', 'logL'])

    srcdata = np.array(out["src"])

    outX = [i.X for i in out["samples"]]
    outY = [height-i.Y for i in out["samples"]]

    plot.plot_histogram(data = outX, bins = width, title = "X_histogram of posterior samples")
    plot.plot_histogram(data = outY, bins = height, title = "Y_histogram of posterior samples")
    plot.show_scatterplot(outX,outY, title= "Scatter plot of posterior samples", height = height, width = width)

    outsrcX = [src.X for src in out["src"]]
    outsrcY = [height - src.Y for src in out["src"]]
    plot.plot_histogram(data = outsrcX, bins = width, title="Xsrc")
    plot.plot_histogram(data = outsrcY, bins = height, title="Ysrc")
    plot.show_scatterplot(outsrcX, outsrcY, title= "Scatter plot of sources", height = height, width = width)


if __name__ == '__main__':
    run_source_detect(mode = "Manual" )
