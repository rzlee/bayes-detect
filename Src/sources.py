import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from math import *
import random
from plot import *
import time
import pickle
import copy
import warnings
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN
import os

Config = {}

config_found = 1

data_map = None

try:
    ConfigFile = open(os.path.dirname(os.path.realpath(__file__))+"\config.cfg", "r")
except IndexError:
    print "Can't find the config file"
    config_found = 0

if config_found==1:
    Lines = ConfigFile.readlines()

    for line in Lines:
        if line[0]=='#' or line[0]=='\n':
            pass
        else:
            try:
                Config[line.split('=')[0]]= line.split('=')[1].rstrip()
            except IndexError:
                pass

         
    File = (Config['IMAGE_PATH'])

    if File[-5:] == '.fits':
        hdulist   = fits.open(File)
        data_map   = (hdulist[0].data)

    if File[-5:] != '.fits':
        s = open(File,'r')
        data_map = pickle.load(s)
        s.close()

if config_found==0:
    
    File = "C:/Users/chaithuzz2/Desktop/Bayes_detect/assets/simulated_images/multinest_toy_noised"

    if File[-5:] == '.fits':
        hdulist   = fits.open(File)
        data_map   = (hdulist[0].data)

    if File[-5:] != '.fits':
        s = open(File,'r')
        data_map = pickle.load(s)
        s.close()


height, width = len(data_map), len(data_map[0])
no_pixels = width*height

#Converting the data_map into a vector for likelihood calculations
data_map = data_map.flatten()


#Useful in likelihood evaluation for calculating the simulated object as the function of indices
x_forcalc = np.arange(0, width)
y_forcalc = np.arange(0, height)
xx, yy = np.meshgrid(x_forcalc, y_forcalc, sparse=True)


class Source:
    
    """
     This is a class which instantiates a source object with its attributes.

    Attributes
    ----------
    X : float
        x coordinate of the center of the object
    Y : float
        y coordinate of the center of the object
    A : float
        Amplitude of the object
    R : float
        Spatial extent of the object
    logL : float 
        Log likelihood of the object
    logWt : float
        Log weight of the object"""
          
    def __init__(self):

        self.X = None
        self.Y = None
        self.A = None
        self.R = None
        self.logL = None
        self.logWt = None


def log_likelihood(Source):

    """
    Returns the log likelihood of the source object.

    Parameters
    ----------
    Source : object
        A source object.

    Returns
    -------
    log likelihood : float
        log likelihood of the input object.

    Raises
    ------
        TypeError : When we pass an object with any of X, Y, A, R attributes as None type

    """           

    simulated_map = Source.A*np.exp(-1*((xx-Source.X)**2+(yy-Source.Y)**2)/(2*(Source.R**2)))
    diff_map = data_map - simulated_map.flatten()
    return -0.5*np.dot(diff_map, np.transpose((1/(noise**2))*diff_map)) - K    
    

def proposed_model(x, y, X, Y, A, R):

    """
    Returns the contribution to a pixel at (x,y) from source [X,Y,A,R]

    Parameters
    ----------
    x : float
        The x coordinate of the pixel of interest.
    y : float
        The y coordinate of the pixel of interest.
    X : float
        The x coordinate of the center of the source under consideration
    Y : float
        The y coordinate of the center of the source under consideration
    A : float
        The value of amplitude of the source under consideration
    R : float
        The value of spatial extent of the source under consideration         

    Returns
    -------
    Intensity value at (x,y)           

    """
    return A*np.exp(((x-X)**2 + (y-Y)**2)/(2*(R**2)))
    

def sample_source():
    
    """
    Sampling the object from prior distribution.

    Returns
    -------
    src : object
        The source object with X,Y,A,R sampled from their prior distribution and log likelihood calculated.

    
    """

    src = Source()
    src.X = random.uniform(0.0, x_upper)
    src.Y = random.uniform(0.0, y_upper) 
    src.A = random.uniform(amplitude_lower, amplitude_upper)
    src.R = random.uniform(R_lower, R_upper)
    src.logL = log_likelihood(src)
    return src


def get_sources(no_active_points):

    """
    Returns an array of source objects sampled from their prior distribution.

    Parameters
    ----------
    no_active_points : int
        The number of source objects to be returned

    Returns
    -------
    src_array : array
        An array of objects with size equal to no_active_points.

    """

    src_array = []
    
    for i in range(no_active_points):
        src_array.append(sample_source())
    
    return src_array


def getPrior_A():

    """
    Returns
    -------
    bounds : tuple
        a tuple of the amplitude bounds.

    """

    bounds = amplitude_lower, amplitude_upper;
    return bounds


def getPrior_R():

    """
    Returns
    -------
    bounds : tuple
        a tuple of the R bounds.

    """

    bounds = R_lower, R_upper; 
    return bounds


def getPrior_X():

    """
    Returns
    -------
    bounds : tuple
        a tuple of the X bounds.

    """

    bounds = 0.0, width;
    return bounds


def getPrior_Y():

    """
    Returns
    -------
    bounds : tuple
        a tuple of the Y bounds.

    """
    
    bounds = 0.0, height;
    return bounds


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


def read(filename):

    """
    Reads an array from a pickle and returns it.

    Parameters
    ----------
    filename : str
        location of the pickle

    Returns
    -------
    data : array
        Pickle data

    """

    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data

#---------------------------------------------------------------------------------------------------------------
#                                     MAIN NESTED SAMPLER CLASS
#---------------------------------------------------------------------------------------------------------------


class Nested_Sampler(object):

    """
    An Implementation of Nested Sampling proposed by John Skilling and Sivia.


    Attributes
    ----------
    no_active_samples : int
        number of active samples
    maximum_iterations : int
        maximum number of iterations
    sample : str
        sampling type
    convergence_threshold : float
        stopping criterion based on evidence
    active_samples : array
        array containing the active samples
    log_evidence : float
        Log evidence
    posterior_inferences : array
        Posterior samples 
    log_width : float
        Log width of the prior
    Information : float
        Information for error estimation in evidence
    no_likelihood : int
        To keep track of number of likelihood evaluations made    


    References 
    ----------
    .. [1] http://www.inference.phy.cam.ac.uk/bayesys/
    .. [2] Shaw, Bridges, Hobson 2007, MNRAS, 378, 1365 

    """
    
    def __init__(self, no_active_samples, max_iter, sample = "metropolis", conv_thresh=0.1):

        """
        Initializes the nested sampler.

        Parameters
        ----------
        no_active_samples : int
            Number of active points which aid in sampling
        max_iter : int
            Maximum number of iterations to run
        sample : str
            Sampling mode
            
            * "uniform" = Samples the points randomly from a uniform distribution.  
            * "metropolis" = Samples the points according to Metropolis principle.
            * "clustered_ellipsoidal" = Samples the points according to Clustered ellipsoidal method.
            
        conv_thresh : float
            Stopping criterion based on the current evidence in an iteration.  
            
        """

 
        self.no_active_samples     = no_active_samples
        self.maximum_iterations    = max_iter
        self.sample                = sample
        self.convergence_threshold = 0.1
        self.active_samples        = get_sources(self.no_active_samples)
        self.log_evidence          = None # Log evidence
        self.posterior_inferences  = []   # Posterior samples 
        self.log_width             = None # Log width of the prior
        self.Information           = None # Information for error estimation in evidence
        self.no_likelihood         = no_active_samples # To keep track of number of likelihood evaluations made

    
    def fit(self):

        """
        Runs the nested sampling procedure.

        Returns
        -------
        A dict mapping the following to their values.

            *  src - Active points
            *  samples - Posterior samples 
            *  logZ - The log evidence
            *  Information - The Information for error estimation
            *  likelihood_calculations - Number of likelihood evaluations
            *  iterations - Number of iterations until stopping

        """

        #Initializing evidence and prior mass
        self.log_evidence = -1e300
        self.log_width = log(1.0 - exp(-1.0 / self.no_active_samples))
        self.Information = 0.0
        LogL = [i.logL for i in self.active_samples]
        iteration = None
        stop = None
        prev_stop = 0.0
       
        for iteration in range(1,60000):
            smallest = 0
            
            #Finding the object with smallest likelihood
            smallest = np.argmin(LogL)
            
            #Assigning local evidence to the smallest sample
            self.active_samples[smallest].logWt = self.log_width + self.active_samples[smallest].logL;
            
            largest = np.argmax(LogL)

            
            #Calculating the updated evidence
            temp_evidence = np.logaddexp(self.log_evidence, self.active_samples[smallest].logWt)
            
            #Calculating the information which will be helpful in calculating the uncertainity
            self.Information = exp(self.active_samples[smallest].logWt - temp_evidence) * self.active_samples[smallest].logL + \
            exp(self.log_evidence - temp_evidence) * (self.Information + self.log_evidence) - temp_evidence;
            
            # FIX ME : Add a stopping criterion condition 

            self.log_evidence = temp_evidence

            stopping = self.active_samples[largest].logL + self.log_width - self.log_evidence 
            #print str(stopping)


            if iteration%100 == 0 or iteration==1:
                print "Iteration: "+str(iteration) + "  maxZ: "+str(stopping)  

            if stopping < self.convergence_threshold:
                if stop == 1:
                    break
            
            if iteration >= self.maximum_iterations:
                if stop == 0:
                    break
            #print str(self.active_samples[smallest].X)+" "+str(self.active_samples[smallest].Y)+" "+str(self.active_samples[smallest].logL)
            
            sample = Source()
            sample.__dict__ = self.active_samples[smallest].__dict__.copy()

            #storing posterior points
            self.posterior_inferences.append(sample)
            
            #New likelihood constraint 
            likelihood_constraint = self.active_samples[smallest].logL

            survivor = int(smallest)

            while True:
                survivor = int(self.no_active_samples * np.random.uniform(0,1)) % self.no_active_samples  # force 0 <= copy < n
                if survivor != smallest:
                    break

            if self.sample == "metropolis":
                #Obtain new sample using Metropolis principle
                updated, number = self.metropolis_sampling(obj = self.active_samples[survivor], LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number

            if self.sample == "clustered_ellipsoidal":
                #Obtain new sample using Clustered ellipsoidal sampling
                updated, number = self.clustered_sampling(active_points = self.active_samples, LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number  

            if self.sample == "uniform":
                #Obtain new sample using uniform sampling principle
                updated, number = self.uniform_sampling(LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number


            #Shrink width  
            self.log_width -= 1.0 / self.no_active_samples;

        # FIX ME: Incorporate the active samples into evidence calculation and information after the loop
        return { "src":self.active_samples,
            "samples":self.posterior_inferences, 
            "logZ":self.log_evidence,
            "Information":self.Information,
            "likelihood_calculations":self.no_likelihood,
            "iterations":self.maximum_iterations 
            }


    def metropolis_sampling(self, obj, LC, likelihood_calc):

        """
        Returns the sample satisfying the likelihood condition by metropolis sampling 

        Parameters
        ----------
        obj : object
            The sample to evolve
        LC  : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        evolved - object
            The evolved sample satisfying the likelihood constraint
        number - int
            The updated likelihood calculations number

        """

        #Instantiating the metropolis sampler object
        Metro = Metropolis_sampler(to_evolve = obj, likelihood_constraint = LC, no =likelihood_calc )
        evolved, number = Metro.sample()
        return evolved, number


    def clustered_sampling(self, active_points, LC, likelihood_calc ):

        """
        Returns the sample satisfying the likelihood condition by clustered ellipsoidal sampling 

        Parameters
        ----------
        active_points : array
            The full set of active points at current state
        LC : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        sample : object
            The evolved sample satisfying the likelihood constraint
        number : int
            The updated likelihood calculations number

        """


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

    
    def uniform_sampling(self, LC, likelihood_calc):

        """
        Returns the sample satisfying the likelihood condition by uniform random sampling 

        Parameters
        ----------
        LC  : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        evolved : object
            The evolved sample satisfying the likelihood constraint
        number : int
            The updated likelihood calculations number

        """

        unif = uniform_sampler(likelihood_constraint = LC, no =likelihood_calc)
        evolved, number = unif.sample()
        return evolved, number      


#---------------------------------------------------------------------------------------------------------------
#                                     UNIFORM SAMPLER
#---------------------------------------------------------------------------------------------------------------



class uniform_sampler(object):

    """
    An Implementation of uniform sampling to randomly pick a sample satisfying the likelihood
    constraint in the current nested sampling phase.

    Attributes
    ----------
    LC : float
        likelihood constraint for the point
    number : int
        likelihood calculations until now        

    """

    def __init__(self, likelihood_constraint, no):

        """
        Initializes the uniform sampler

        Parameters
        ----------
        likelihood_constraint: float
            name says it all
        no : int
            Number of likelihood evaluations until this point

        """

        self.LC     = likelihood_constraint
        self.number = no
                
    
    def sample(self):

        """
        Method to pick the sample satisfying the likelihood constraint using uniform sampling

        Returns
        -------
        new : object
            The evolved sample  
        number : int
            Number of likelihood calculations after sampling  

        """

        new   = Source()
                        
        x_l, x_u = getPrior_X()
        y_l, y_u = getPrior_Y()
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A()

        while(True):
            
            new.X = np.random.uniform(x_l,x_u)
            new.Y = np.random.uniform(y_l,y_u)
            new.A = np.random.uniform(a_l,a_u)
            new.R = np.random.uniform(r_l,r_u)
            new.logL = log_likelihood(new)
            self.number+=1
            
            if(new.logL > self.LC):
                break
                        
        return new, self.number

#---------------------------------------------------------------------------------------------------------------
#                                     METROPOLIS SAMPLER
#---------------------------------------------------------------------------------------------------------------


class Metropolis_sampler(object):

    """
    An Implementation of Metropolis sampling to pick a sample satisfying the likelihood
    constraint in the current nested sampling phase.  

    Attributes
    ----------
    source : object
       object to evolve
    LC : float
        likelihood constraint for the point
    step : float
        dispersion of the gaussian proposal distribution
    number : int
        likelihood calculations until now
    
    """

    def __init__(self, to_evolve, likelihood_constraint, no):

        """
        Initializes the Metropolis sampler

        Parameters
        ----------
        to_evolve : object
            The sample to evolve
        likelihood_constraint: float
            name says it all
        no : int
            Number of likelihood evaluations until this point
        
        """

        self.source = to_evolve
        self.LC     = likelihood_constraint
        self.step   = dispersion
        self.number = no
                
    
    def sample(self):

        """
        Method to pick the sample satisfying the likelihood constraint using metropolis sampling

        Returns
        -------
        metro : object
            The evolved sample  
        number : int
            Number of likelihood calculations until now  

        """

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


#---------------------------------------------------------------------------------------------------------------
#                                     CLUSTERED ELLIPSOIDAL SAMPLER
#---------------------------------------------------------------------------------------------------------------



class Clustered_Sampler(object):

    """
    Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
    an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
    and Hobson(2008). This method is closer in spirit to the recursive technique advocated in Shaw
    et al.

    Attributes
    ----------
    points : array
        copy of active samples in the current nested sampling phase
    LC : float
        likelihood_constraint
    enlargement : float
        Enlargement factor used in constructing ellipsoids
    clustered_point_set : array
        The points belonging to different clusters
    number_of_clusters : int
        Number of clusters
    activepoint_set : float
        array of activepoint_set for clustering
    ellipsoid_set : array
        array of optimal ellipsoids constructed around the active point set.
    total_vol : float
        Total volume enclosed by the optimal ellipsoids
    number : float
        Number of likelihood evaluations until now 

    References
    ----------
    .. [1] Multinest paper by Feroz and Hobson 2008.
    .. [2] Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)
    .. [3] A Nested Sampling Algorithm for Cosmological Model Selection(2006) Pia Mukherjee , David Parkinson , and Andrew R. Liddle
    .. [4] http://www.sjsu.edu/faculty/watkins/ellipsoid.htm 

    """


    def __init__(self, active_samples, likelihood_constraint,enlargement, no):

        """
        Initializes the clustered ellipsoidal sampler.

        Parameters
        ----------
        active_samples : array
            The array containing the active samples for this clustered sampling phase
        likelihood_constraint : float
            Name says it all
        enlargement_factor : float
            The enlargement factor for ellipsoids
        no : int
            Number of likelihood calculations until the current sampling phase  

        """

        self.points = copy.deepcopy(active_samples)
        self.LC = likelihood_constraint
        self.enlargement = 1.5
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.activepoint_set = self.build_set()
        self.ellipsoid_set = self.optimal_ellipsoids()
        self.total_vol = None
        self.number = no

    
    def build_set(self):

        """
        Builds set of the active samples in array format for clustering.

        Returns
        -------
        array : array
            Array containing the active samples. Each element [X,Y] represents an object 

        """

        array = []
        for active_sample in self.points:
            array.append([float(active_sample.X), float(active_sample.Y)])
        return np.array(array)            
        

    def cluster(self, activepoint_set):

        """ 
        Clusters an array of samples using kmeans2

        Parameters
        ----------
        activepoint_set : array
            Set of active points to be clustered

        Returns
        -------
        number_of_clusters : int
            name says it all
        labels : array
            Cluster labels assigned by kmeans for each sample
        activepoint_set : array
            The point set       
        
        """
        
        db = DBSCAN(eps=10, min_samples=10).fit(activepoint_set)
        labels = db.labels_
        number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #print str(labels)
        #centroid, labels = kmeans2(activepoint_set, 5)
        #number_of_clusters = len(centroid)  
        return number_of_clusters, labels, activepoint_set    


    def optimal_ellipsoids(self):

        """

        Method for calculating ellipsoids around the individual clusters.

        Returns
        -------
        ellipsoids : array
            An array of ellipsoids to sample from.  

        """


        self.number_of_clusters, point_labels, pointset = self.cluster(self.activepoint_set)#Cluster and find centroids
        clust_points = np.empty(self.number_of_clusters,dtype=object)
        ellipsoids = np.empty(self.number_of_clusters,dtype=object)
        for i in range(self.number_of_clusters):
            clust_points[i] = np.array(pointset[np.where(point_labels==i)])
        invalid = []    
        for i in range(self.number_of_clusters):
            if len(clust_points[i]) > 1:
                try:
                    ellipsoids[i] = Ellipsoid(points=clust_points[i],
                              enlargement_factor = 2.0)#enlargement*np.sqrt(len(self.activepoint_set)/len(clust_points[i]))
                except np.linalg.linalg.LinAlgError:
                    ellipsoids[i] = None
                    #print str(i)
                    invalid.append(i)
            else:
                ellipsoids[i] = None
                #print str(i)
                invalid.append(i)
        ellipsoids = np.delete(ellipsoids, invalid)
        #print len(ellipsoids)         
        return ellipsoids


    def recursive_bounding_ellipsoids(self, data, ellipsoid=None):

        """
        Implementation of finding minimum bounding ellipsoids recursively. (work in progress) 

        Parameters
        ----------
        data : array
            The active samples around which ellipsoids are to be built.
        ellipsoid : object
            Starting ellipsiod. None by default

        Returns
        -------
        ellipsoids : array
           Array of ellipsoids satisfying all the conditions               

        """

        ellipsoids = []
        if ellipsoid is None:
            ellipsoid = Ellipsoid(points = data, enlargement_factor=1.0)
        centroids, labels = kmeans2(data, 2, iter=10)
        clustered_data = [None, None]
        clustered_data[0] = [data[i] for i in range(len(data)) if labels[i]==0]
        clustered_data[1] = [data[i] for i in range(len(data)) if labels[i]==1]
        vol = [0.0,0.0]
        clustered_ellipsoids = np.empty(2,object)  
        for i in [0, 1]:
            if(len(clustered_data[i]) <= 1):
                clustered_ellipsoids[i] = Ellipsoid(clustered_data[i],1.0)
                vol[i]= clustered_ellipsoids[i].volume 
        do = True

        if(vol[0] + vol[1] < ellipsoid.volume ):
            for i in [0, 1]:
                if(vol[i]>0.0):
                    ellipsoids.extend(self.recursive_bounding_ellipsoids(np.array(clustered_data[i]), clustered_ellipsoids[i]))
        else:
            ellipsoids.append(ellipsoid)
        print len(ellipsoids)    
        return ellipsoids    

     
    def sample(self):

        """
        Sampling from the built ellipsoids.

        Returns
        -------
        clust : object
            The sampled point satisfying the likelihood constraint
        number : int
            The number of likelihood calculations until this

        """

        #vols = np.array([len(i.clpoints) for i in self.ellipsoid_set])
        #print str(vols)
        #print str(np.sum(vols))
        #vols = vols/np.sum(vols)
        #print str(np.sum(vols))
        arbit = np.random.uniform(0,1)
        trial = Source()
        clust = Source()
        z = int((len(self.ellipsoid_set))*arbit)
        #z = None
        #for i in range(len(vols)):
        #    if(arbit<=vols[i]):
        #        z = i
        #        break
        #print "Sampling from ellipsoid : "+ str(z)
        #print str(z)        
        points = None
        try:
            points = self.ellipsoid_set[z].sample(n_points=50)
        except IndexError:
            print "\n"
            print "\n"
            print "Please adjust the clustering parameters and try again."
            print "\n"
            print "\n"            
        max_likelihood = self.LC
        #print "likelihood_constraint: "+str(max_likelihood)
        count = 0
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A()
        while count<50:
            trial.X = points[count][0]
            trial.Y = points[count][1]
            trial.A = np.random.uniform(a_l,a_u)
            trial.R = np.random.uniform(r_l,r_u)            
            trial.logL = log_likelihood(trial)
            #print "Trial likelihood for point"+" "+str(count)+": "+str(trial.logL)
            self.number+=1

            if(trial.logL > max_likelihood):
                clust.__dict__ = trial.__dict__.copy()
                max_likelihood = trial.logL
                break
                                
            count+=1
        #if(clust.logL > self.LC):
            #print "Found the point with likelihood greater than : "+ str(self.LC) 
        
        return clust,self.number     
             

#---------------------------------------------------------------------------------------------------------------
#                                     ELLIPSOID CLASS
#---------------------------------------------------------------------------------------------------------------

       
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

    def __init__(self, points, enlargement_factor):

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
        #self.volume = self.find_volume()
        

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
        x_l, x_u = getPrior_X()
        y_l, y_u = getPrior_Y()
        r_l, r_u = getPrior_R()
        a_l, a_u = getPrior_A()        
        #print str(values)
        scaled = np.dot(vects, np.diag(np.sqrt(np.absolute(values))))
        #print str(scaled)
        bord = 1
        new = None    
        for i in range(n_points):
            #count = 0
            while bord==1:
                bord = 0
                randpt = np.random.randn(dim)
                point  = randpt* np.random.rand()**(1./dim) / np.sqrt(np.sum(randpt**2))
                new =  np.dot(scaled, point) + self.centroid

                #print str(new)

                if(new[0] > x_u or new[0] < x_l): bord = 1;
                if(new[1] > y_u or new[1] < y_l): bord = 1;
                #count+=1
                #if(count >=200): new = self.centroid

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
    
    global amplitude_upper
    global amplitude_lower
    global x_upper
    global y_upper
    global R_upper
    global R_lower
    global noise
    global K
    global dispersion
    global output_loc
    global stop 

    if mode == "ipython":
        dispersion = disp
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
        output_loc = 'C:\Users\chaithuzz2\Desktop\Bayes_detect\output\samples.dat'
        stop = 0

    if mode == "Manual":
        dispersion = float(Config['DISPERSION'])
        amplitude_upper = float(Config['A_PRIOR_UPPER'])
        amplitude_lower = float(Config['A_PRIOR_LOWER'])
        # FIX : Don't forget to change
        x_upper = float(Config['X_PRIOR_UPPER'])
        y_upper = float(Config['Y_PRIOR_UPPER'])
        R_upper = float(Config['R_PRIOR_UPPER'])
        R_lower = float(Config['R_PRIOR_LOWER'])
        noise   = float(Config['NOISE'])
        K = (no_pixels/2)*(np.log(2*np.pi) + 4*np.log(abs(noise)))
        max_iter = int(Config['MAX_ITER'])
        n = int(Config['ACTIVE_POINTS'])
        sample_type = str(Config['SAMPLER'])
        output_loc = str(Config['OUTPUT_DATA_PATH'])
        stop = int(Config['STOP_BY_EVIDENCE'])
    
    print output_loc 
    nested = Nested_Sampler(no_active_samples = n, max_iter = max_iter, sample = sample_type)
    out  = nested.fit()

    elapsedTime = time.time() - startTime
    print "elapsed time: "+str(elapsedTime) 
    print "log evidence: "+str(out["logZ"])
    print "number of iterations: "+str(out["iterations"])
    print "likelihood calculations: "+str(out["likelihood_calculations"])

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

    plot_histogram(data = outX, bins = width, title = "X_histogram of posterior samples")
    plot_histogram(data = outY, bins = height, title = "Y_histogram of posterior samples")
    show_scatterplot(outX,outY, title= "Scatter plot of posterior samples", height = height, width = width)

    outsrcX = [i.X for i in out["src"]]
    outsrcY = [height-i.Y for i in out["src"]]
    plot_histogram(data = outsrcX, bins = width, title="Xsrc")
    plot_histogram(data = outsrcY, bins = height, title="Ysrc")
    show_scatterplot(outsrcX,outsrcY, title= "Scatter plot of sources", height = height, width = width)


if __name__ == '__main__':
    run_source_detect(mode = "Manual" )