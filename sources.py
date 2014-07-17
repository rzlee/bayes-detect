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
from plot import *
import time
import pickle

"""Reading the Image data from fits file"""
"""fitsFile = "simulated_images/multinest_toy"

hdulist   = fits.open(fitsFile)
data_map   = (hdulist[0].data)"""

File = "simulated_images/multinest_toy_noised"

s = open(File,'r')
data_map = pickle.load(s)
s.close()

height, width = len(data_map), len(data_map[0])
no_pixels = width*height

"""Converting the data_map into a vector for likelihood calculations"""
data_map = data_map.flatten()


"""Useful in likelihood evaluation for calculating the simulated object as the function of indices"""
x_forcalc = np.arange(0, width)
y_forcalc = np.arange(0, height)
xx, yy = np.meshgrid(x_forcalc, y_forcalc, sparse=True)


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






"""Nested sampler main class"""
class Nested_Sampler(object):


    """Initialization for the Nested_Sampler"""

    def __init__(self, no_active_samples, max_iter, sample = "metropolis", plot=False, conv_thresh=0.1):
 
        """Number of active_samples in the nested sampling loop to start"""
        self.no_active_samples     = no_active_samples

        """Maximum number of iterations after which the loop is terminated"""
        self.maximum_iterations    = max_iter
        
        """ The sampling method used to draw a new sample satisfying the likelihood constraint"""
        self.sample                = sample

        """Plot posterior samples while running the loop to check how the method is working"""
        self.plot                  = plot

        """Stopping criterion for nested sample. The same value used in the Multinest paper """
        self.convergence_threshold = 0.1

        """ Sampling active points from the prior distribution specified """
        self.active_samples        = get_sources(self.no_active_samples)
        
        """ Variable to hold evidence at each iteration"""
        self.log_evidence              = None

        """ Posterior samples for evidence and plotting """
        self.posterior_inferences  = []

        """ Prior mass which is used to calculate the weight of the point at each iteration"""
        self.log_width             = None

        """ Information for calculating the uncertainity of calculating the evidence """
        self.Information           = None

        """ Total number of likelihood calculations""" 
        self.no_likelihood         = no_active_samples

    
    """ Method that runs the main nested sampling loop"""
    
    def fit(self):

        """ Initializing evidence and prior mass """
        self.log_evidence = -1e300
        self.log_width = log(1.0 - exp(-1.0 / self.no_active_samples))
        self.Information = 0.0
        LogL = [i.logL for i in self.active_samples]
        iteration = None
        stop = None
        prev_stop = 0.0
       
        for iteration in range(1,self.maximum_iterations):
            smallest = 0
            """Finding the object with smallest likelihood"""
            smallest = np.argmin(LogL)
            """Assigning local evidence to the smallest sample"""
            self.active_samples[smallest].logWt = self.log_width + self.active_samples[smallest].logL;
            
            largest = np.argmax(LogL)

            stop = self.active_samples[largest].logL + self.log_width - self.log_evidence

            if iteration%1000 == 0 or iteration==1:
                print str(iteration)
            
            """Calculating the updated evidence"""
            temp_evidence = np.logaddexp(self.log_evidence, self.active_samples[smallest].logWt)
            
            """Calculating the information which will be helpful in calculating the uncertainity"""
            self.Information = exp(self.active_samples[smallest].logWt - temp_evidence) * self.active_samples[smallest].logL + \
            exp(self.log_evidence - temp_evidence) * (self.Information + self.log_evidence) - temp_evidence;
            
            # FIX ME : Add a stopping criterion condition 

            self.log_evidence = temp_evidence

            #print str(self.active_samples[smallest].X)+" "+str(self.active_samples[smallest].Y)+" "+str(self.active_samples[smallest].logL)
            
            sample = Source()
            sample.__dict__ = self.active_samples[smallest].__dict__.copy()

            """storing posterior points"""
            self.posterior_inferences.append(sample)
            
            """New likelihood constraint""" 
            likelihood_constraint = self.active_samples[smallest].logL

            survivor = int(smallest)

            while True:
                survivor = int(self.no_active_samples * np.random.uniform(0,1)) % self.no_active_samples  # force 0 <= copy < n
                if survivor != smallest:
                    break

            if self.sample == "metropolis":
                """Obtain new sample using Metropolis principle"""
                updated, number = self.metropolis_sampling(obj = self.active_samples[survivor], LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number

            if self.sample == "clustered_ellipsoidal":
                """Obtain new sample using Clustered ellipsoidal sampling"""
                updated, number = self.clustered_sampling(active_points = self.active_samples, LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number  

            """Shrink width"""  
            self.log_width -= 1.0 / self.no_active_samples;

        # FIX ME: Incorporate the active samples into evidence calculation and information after the loop """
        return { "src":self.active_samples,
            "samples":self.posterior_inferences, 
            "logZ":self.log_evidence,
            "Information":self.Information,
            "likelihood_calculations":self.no_likelihood,
            "iterations":self.maximum_iterations 
            }


    """ Method for drawing a new sample using the metropolis hastings principle """  

    def metropolis_sampling(self, obj, LC, likelihood_calc):
        "Instantiating the metropolis sampler object"
        Metro = Metropolis_sampler(to_evolve = obj, likelihood_constraint = LC, no =likelihood_calc )
        evolved, number = Metro.sample()
        return evolved, number


    """ Method for drawing a new sample using clustered ellipsoidal sampling"""
    
    def clustered_sampling(self, active_points, LC, likelihood_calc ):
        Clust = Clustered_Sampler(active_samples=active_points, likelihood_constraint=LC, enlargement=1.0, no=likelihood_calc)
        sample = None
        number = None
        while True:
            sample, number = Clust.sample()
            if(sample.logL > LC):
                #print "In nest: found from clustered sampling"
                break
            Clust = Clustered_Sampler(active_samples=active_points, likelihood_constraint=LC, enlargement=1.0, no=number)   
        return sample, number





"""Metropolis sampler"""
class Metropolis_sampler(object):

    """Initializing metropolis sampler"""

    def __init__(self, to_evolve, likelihood_constraint, no):

        self.source = to_evolve
        self.LC     = likelihood_constraint
        self.step   = 8.0
        self.number = no
                
    """Sampling from the prior subject to constraints according to Metropolis method 
    proposed by Sivia et al discussed in Multinest paper by feroz and Hobson"""

    def sample(self):
        metro = Source()
        metro.__dict__ = self.source.__dict__.copy()
        start = Source()
        start.__dict__ = self.source.__dict__.copy()
        new   = Source()
        self.number+=1
        count = 0
        hit = 0
        miss = 0
        
        x_l, x_u = getPrior_X()
        y_l, y_u = getPrior_Y()
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A()

        stepnormalize = self.step/x_u

        stepX    = self.step
        stepY    = stepnormalize*(y_u-y_l)
        stepA    = stepnormalize*(a_u - a_l)
        stepR    = stepnormalize*(r_u-r_l)        
        
        bord = 1

        while(count<20):
            
            while bord==1:
                bord = 0
                new.X    = metro.X + stepX * (2.*np.random.uniform(0, 1) - 1.);
                new.Y    = metro.Y + stepY * (2.*np.random.uniform(0, 1) - 1.);
                new.A    = metro.A + stepA * (2.*np.random.uniform(0, 1) - 1.);
                new.R    = metro.R + stepR * (2.*np.random.uniform(0, 1) - 1.);

                if(new.X > x_u or new.X < x_l): bord = 1;
                if(new.Y > y_u or new.Y < y_l): bord = 1;
                if(new.A > a_u or new.A < a_l): bord = 1;
                if(new.R > r_u or new.R < r_l): bord = 1;                

            new.logL = log_likelihood(new)
            self.number+=1
            
            if(new.logL > self.LC):
                metro.__dict__ = new.__dict__.copy()
                hit+=1
            else:
                miss+=1
            
            if( hit > miss ):   self.step *= exp(1.0 / hit);
            if( hit < miss ):   self.step /= exp(1.0 / miss);

            stepnormalize = self.step/x_u         

            stepX    = self.step
            stepY    = stepnormalize*(y_u-y_l)
            stepA    = stepnormalize*(a_u - a_l)
            stepR    = stepnormalize*(r_u-r_l)        
        
            
            count+=1
            bord=1
                    
        return metro, self.number





"""Main method to start nested sampling"""
def run_source_detect(samples, iterations, sample_method, prior,noise_rms):
    startTime = time.time()
    
    global amplitude_upper
    global amplitude_lower
    global x_upper
    global y_upper
    global R_upper
    global R_lower
    global noise
    global K


    amplitude_upper = prior[2][1]
    amplitude_lower = prior[2][0]

    x_upper = prior[0][1]
    y_upper = prior[1][1]

    R_upper = prior[3][1]
    R_lower = prior[3][0]
    
    noise   = noise_rms  

    K = (no_pixels/2)*(np.log(2*np.pi) + 4*np.log(abs(noise)))

    n = samples
    max_iter = iterations
    sample_type = sample_method
    
    nested = Nested_Sampler(no_active_samples = samples, max_iter = max_iter, sample = sample_type)
    out  = nested.fit()

    elapsedTime = time.time() - startTime
    print "elapsed time: "+str(elapsedTime) 
    print "log evidence: "+str(out["logZ"])
    print "number of iterations: "+str(out["iterations"])
    print "likelihood calculations: "+str(out["likelihood_calculations"])

    data = np.array(out["samples"])
    srcdata = np.array(out["src"])
    outX = [i.X for i in out["samples"]]
    outY = [200-i.Y for i in out["samples"]]   

    plot_histogram(data = outX, bins = 200, title = "X_histogram of posterior samples")
    plot_histogram(data = outY, bins = 200, title = "Y_histogram of posterior samples")
    show_scatterplot(outX,outY, title= "Scatter plot of posterior samples", height = height, width = width)

    outsrcX = [i.X for i in out["src"]]
    outsrcY = [200-i.Y for i in out["src"]]
    plot_histogram(data = outsrcX, bins = 200, title="Xsrc")
    plot_histogram(data = outsrcY, bins =200, title="Ysrc")
    show_scatterplot(outsrcX,outsrcY, title= "Scatter plot of sources", height = height, width = width)

if __name__ == '__main__':
    prior_array = [[0.0,200.0],[0.0,200.0],[1.0,12.5],[2.0,9.0]]
    noise = 2.0
    run_source_detect(samples = 400, iterations = 8000, sample_method = "metropolis", prior = prior_array, noise_rms = noise)

           
         


