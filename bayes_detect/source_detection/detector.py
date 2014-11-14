import numpy as np
import time
import pickle
import plot
import datetime

from astropy.io import fits
from astropy.io import ascii

from nested import Nested_Sampler

def load_config(path):
    """
    Read the config file and return relevant information
    Parameters
    __________
    path : str
        Absolute file path to the config file

    Returns
    ______
        Config : dict
            The config file in the form of a dictionary
        data_map : array
            Array that represents the image
        height : int
            Height of the image
        width : int
            Width of the image
    """
    with open(path) as config_file:
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

    (height, width) = data_map.shape
    data_map = data_map.flatten()
    return (Config, data_map, height, width)

def visualize(out, height, width):
    """
    outX = [i.X for i in out["samples"]]
    outY = [height-i.Y for i in out["samples"]]

    plot.plot_histogram(data = outX, bins = width, title = "X_histogram of posterior samples")
    plot.plot_histogram(data = outY, bins = height, title = "Y_histogram of posterior samples")
    plot.show_scatterplot(outX,outY, title= "Scatter plot of posterior samples", height = height, width = width)
    """

    outsrcX = [src.X for src in out["src"]]
    outsrcY = [height - src.Y for src in out["src"]]
    plot.plot_histogram(data = outsrcX, bins = width, title="Xsrc")
    plot.plot_histogram(data = outsrcY, bins = height, title="Ysrc")
    plot.show_scatterplot(outsrcX, outsrcY, title= "Scatter plot of sources", height = height, width = width)



def manual_source_detection(path, show_plot = True):
    """
    pass in the path to the config file and it'll run the detector
    according to the requested settings
    show_plot defaults to true, set to false for terminal usage
    """
    
    (config, data_map, height, width) = load_config(path) 
    prior = [[0, float(config['X_PRIOR_UPPER'])],
             [0, float(config['Y_PRIOR_UPPER'])],
             [float(config['R_PRIOR_LOWER']), float(config['R_PRIOR_UPPER'])],
             [float(config['A_PRIOR_LOWER']), float(config['A_PRIOR_UPPER'])]]

    sample_params = dict()
    sample_params['type'] = config['SAMPLER']
    if sample_params['type'] == "metropolis":
        sample_params['disp'] = config['DISPERSION']
    if sample_params['type'] == "clustered_sampler":
        sample_params['minPts'] = int(config['MINPTS'])
        sample_params['eps'] = int(config['EPS'])
        sample_params['wait'] = int(config['WAIT'])

    run_source_detect(data_map = data_map, height = height, width = width,
                     active_samples = int(config['ACTIVE_POINTS']),
                     iterations = int(config['MAX_ITER']), sample_params = sample_params,
                     prior = prior, noise_rms = float(config['NOISE']),
                     filepath = config['OUTPUT_DATA_PATH'],
                     stop_by_evidence = config['STOP_BY_EVIDENCE'],
                     show_plot = show_plot, write = True)


def run_source_detect(data_map = None, height = -1, width = -2, active_samples = None,
                      iterations = None, sample_params = None, prior = None,
                      noise_rms = None, filepath = None, stop_by_evidence = True,
                      show_plot = True, write = False):

    if len(data_map.shape) != 1:
        #it needs to be flattened
        data_map = data_map.flatten()

    #some error catching
    if "metropolis" in sample_params and "disp" not in sample_params:
        #we don't have a default dispersion value, just throw an error
        raise Exception("Metropolis Hastings sampler selected without a dispersion value")

    params = dict()
    if "disp" in sample_params:
        #only in metropolis hastings
        if sample_params['disp'] <= 0:
            raise Exception("Dispersion value must be non negative")

        params['dispersion'] = sample_params['disp']

    if sample_params['type'] == "clustered_sampler":
        if "minPts" not in sample_params:
           raise Exception("minPts value missing")
        if "eps" not in sample_params:
           raise Exception("eps value missing")
        if sample_params['wait'] < 0:
            raise Exception("invalid wait period")

        params['eps'] = sample_params['eps'] 
        params['minPts'] = sample_params['minPts']
        params['wait'] = sample_params['wait']

    no_pixels = height * width

    params['amp_upper'] = prior[2][1] 
    params['amp_lower'] = prior[2][0] 
    params['x_upper'] = prior[0][1] 
    params['y_upper'] = prior[1][1] 
    params['r_upper'] = prior[3][1] 
    params['r_lower'] = prior[3][0] 
    params['noise'] = noise_rms
    params['k'] = (no_pixels/2)*(np.log(2*np.pi) + 4*np.log(abs(noise_rms)))
    params['n'] = active_samples
    params['max_iter'] = iterations
    params['type'] = sample_params['type'] 
    params['output_loc'] = filepath #this must be an existing file/folder
    params['stop_by_evidence'] = stop_by_evidence
    params['width'] = width
    params['height'] = height

    #some clever features
    
    #if iterations <= 0, then we obviously want to stop by evidence
    if iterations <= 0:
        params['stop_by_evidence'] = True
    #if filepath is None, then we obviously dont want to write anything
    if filepath == None:
        write = False

    startTime = time.time()

    nested = Nested_Sampler(data_map, params, sampler = params['type'])
    out  = nested.fit()

    elapsedTime = time.time() - startTime

    print "elapsed time: " + str(datetime.timedelta(seconds=elapsedTime))
    print "log evidence: " + str(out["logZ"])
    print "number of iterations: " + str(out["iterations"])
    print "likelihood calculations: " + str(out["likelihood_calculations"])

    data = np.array(out["samples"])

    X = np.array([i.X for i in data])
    Y = np.array([i.Y for i in data])
    A = np.array([i.A for i in data])
    R = np.array([i.R for i in data])
    logL = np.array([i.logL for i in data])

    if write:
        ascii.write([X, Y, A, R, logL], params['output_loc'],
                     names=['X', 'Y', 'A', 'R', 'logL'])

    if show_plot:
        visualize(out, height, width)
