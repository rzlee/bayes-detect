"""Implementation of clustered ellipsoidal method for using in multimodal nested sampling as
an improvement to detect modes in the posterior. This was proposed in multinest paper by Feroz
and Hobson(2008). This method is closer in spirit to the recursive technique advocated in Shaw
et al. 

References :
===========

Multinest paper by Feroz and Hobson 2008.
Shaw R., Bridges M., Hobson M.P., 2007, MNRAS, in press (astro-ph/0701867)
A Nested Sampling Algorithm for Cosmological Model Selection(2006) Pia Mukherjee , David Parkinson , and Andrew R. Liddle
http://www.sjsu.edu/faculty/watkins/ellipsoid.htm 

"""

import numpy as np
import math
#from sources import *
import random
import copy
from sklearn.cluster import AffinityPropagation
import warnings
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from pylab import figure,show

class Clustered_Sampler(object):

    """Initialize using the information of current active samples and the object to evolve"""
    def __init__(self, active_samples, likelihood_constraint,enlargement, no):#

        self.points = copy.deepcopy(active_samples)
        self.LC = likelihood_constraint
        self.enlargement = 1.5
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.activepoint_set = self.points#self.build_set()
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
        centroid, labels = kmeans2(activepoint_set, 10)
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
                              enlargement_factor = 1.0)#enlargement*np.sqrt(len(self.activepoint_set)/len(clust_points[i]))
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
        vols = vols/np.sum(vols)
        arbit = np.random.uniform(0,1)
        trial = Source()
        clust = Source()
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
        #print str(values)
        scaled = np.dot(vects, np.diag(np.sqrt(np.absolute(values))))
        #print str(scaled)
        bord = 1
        new = None    
        for i in range(n_points):
            #count = 0
            randpt = np.random.randn(dim)
            point  = randpt* np.random.rand()**(1./dim) / np.sqrt(np.sum(randpt**2))
            new =  np.dot(scaled, point) + self.centroid
     
            points[i, :] = copy.deepcopy(new)
        return points

    def find_volume(self):
        return (np.pi**2)*(np.sqrt(np.linalg.det(self.covariance_matrix)))/2.

        
        


def show_minimum_bounding_ellipsoids(Xr, with_sampled = False):
    clustellip = Clustered_Sampler(active_samples = Xr,likelihood_constraint=0.0, enlargement= 1.0,no=1)
    ellipsoids = clustellip.ellipsoid_set
    X0 = [i[0] for i in Xr]
    Y0 = [i[1] for i in Xr] 
    plt.figure()
    ax = plt.gca()
    points = []
    #print len(ellipsoids)
    for i in range(len(ellipsoids)):
        if ellipsoids[i]!=None:
            a, b = np.linalg.eig(ellipsoids[i].covariance_matrix)
            c = np.dot(b, np.diag(np.sqrt(a)))
            width = np.sqrt(np.sum(c[:,1]**2)) * 2.
            height = np.sqrt(np.sum(c[:,0]**2)) * 2.
            angle = math.atan(c[1,1] / c[0,1]) * 180./math.pi
            points = ellipsoids[i].sample(n_points = 50)
            ellipse = Ellipse(ellipsoids[i].centroid, width, height, angle)
            ellipse.set_facecolor('None')
            ax.add_patch(ellipse)
    X = [i[0] for i in points]
    Y = [i[1] for i in points]
    if with_sampled == True:
        plt.plot(X, Y, 'bo')
    plt.plot(X0, Y0, 'ro')
    plt.show()
     


