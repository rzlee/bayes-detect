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
import clust_ellip
import copy
from sklearn.cluster import AffinityPropagation
import warnings
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans2

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

            if iteration%100 == 0 or iteration==1:
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
        self.step   = dispersion
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




class Clustered_Sampler(object):

    """Initialize using the information of current active samples and the object to evolve"""
    def __init__(self, active_samples, likelihood_constraint,enlargement, no):#

        self.points = copy.deepcopy(active_samples)
        self.LC = likelihood_constraint
        self.enlargement = 1.5
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.activepoint_set = self.build_set()
        self.ellipsoid_set = self.optimal_ellipsoids()
        self.total_vol = None
        #self.found = False
        self.number = no

    def build_set(self):
        array = []
        for active_sample in self.points:
            array.append([float(active_sample.X), float(active_sample.Y)])
        return np.array(array)            
        

    # FIX ME : This method clusters the samples and return the mean points of each cluster. We will attempt
    #to do agglomerative clustering as an improvement in coming days""" 

    def cluster(self, activepoint_set):
        """af = AffinityPropagation().fit(activepoint_set)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        print str(labels)
        number_of_clusters= len(cluster_centers_indices)
        print "number_of_clusters: "+str(number_of_clusters)"""
        """db = DBSCAN(eps=10, min_samples=4).fit(activepoint_set)
        core_samples = db.core_sample_indices_
        labels = db.labels_
        #print str(labels)
        # Number of clusters in labels, ignoring noise if present.
        number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)"""
        centroid, labels = kmeans2(activepoint_set, 5)
        #print str(len(centroid))
        number_of_clusters = len(centroid)  
        return number_of_clusters, labels, activepoint_set    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""

    def optimal_ellipsoids(self):
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
                              enlargement_factor = 1.5)#enlargement*np.sqrt(len(self.activepoint_set)/len(clust_points[i]))
                except np.linalg.linalg.LinAlgError:
                    ellipsoids[i] = None
                    print str(i)
                    invalid.append(i)
            else:
                ellipsoids[i] = None
                print str(i)
                invalid.append(i)
        ellipsoids = np.delete(ellipsoids, invalid)
        print len(ellipsoids)         
        return ellipsoids


    def recursive_bounding_ellipsoids(self, data, ellipsoid=None):
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

     

    
    """This method is responsible for sampling from the enlarged ellipsoids with certain probability
    The method also checks if any ellipsoids are overlapping and behaves accordingly """
    def sample(self):
        vols = np.array([i.volume for i in self.ellipsoid_set])
        vols = vols/np.max(vols)
        arbit = np.random.uniform(0,1)
        trial = Source()
        clust = Source()
        z =None
        for i in range(len(vols)):
            if(arbit<=vols[i]):
                z = i
                break
        #print "Sampling from ellipsoid : "+ str(z)        
        points = self.ellipsoid_set[z].sample(n_points=50)
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
             



        


"""Class for ellipsoids"""

class Ellipsoid(object):

    def __init__(self, points, enlargement_factor):

        self.clpoints = points
        self.centroid = np.mean(points,axis=0)
        self.enlargement_factor = enlargement_factor
        self.covariance_matrix = self.build_cov(self.centroid, self.clpoints)
        self.inv_cov_mat = np.linalg.inv(self.covariance_matrix)
        self.volume = self.find_volume()
        

    
    def build_cov(self, center, clpoints):
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
        return (np.pi**2)*(np.sqrt(np.linalg.det(self.covariance_matrix)))/2.



"""Main method to start nested sampling"""
def run_source_detect(samples, iterations, sample_method, prior,noise_rms, disp):
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
    prior_array = [[0.0,200.0],[0.0,200.0],[1.0,12.5],[2.0,9.0]]
    noise = 2.0
    run_source_detect(samples = 400, iterations = 8000, sample_method = "clustered_ellipsoidal", prior = prior_array, noise_rms = noise, disp = 4.0)

           
         


