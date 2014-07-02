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
from math import *
from sources import *
import random
import copy
from sklearn.cluster import AffinityPropagation


class Clustered_Sampler(object):

    """Initialize using the information of current active samples and the object to evolve"""
    def __init__(self, active_samples, likelihood_constraint, enlargement, no):

        self.points = copy.deepcopy(active_samples)
        self.LC = likelihood_constraint
        self.enlargement = enlargement
        self.clustered_point_set = None
        self.number_of_clusters = None
        self.activepoint_set = self.build_set()
        self.ellipsoid_set = self.optimal_ellipsoids()
        self.total_vol = None
        self.found = False
        self.number = no

    def build_set(self):
        array = []
        for active_sample in self.points:
            array.append([float(active_sample.X), float(active_sample.Y),float(active_sample.A), float(active_sample.R)])
        return np.array(array)            
        

    # FIX ME : This method clusters the samples and return the mean points of each cluster. We will attempt
    #to do agglomerative clustering as an improvement in coming days""" 

    def cluster(self, activepoint_set):
        af = AffinityPropagation().fit(activepoint_set)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        print str(labels)
        number_of_clusters= len(cluster_centers_indices)
        print str(number_of_clusters)
        return number_of_clusters, labels, activepoint_set    


    """This method builds ellipsoids enlarged by a factor, around each cluster of active samples
     from which we sample to evolve using the likelihood_constraint"""

    def optimal_ellipsoids(self):
        self.number_of_clusters, point_labels, pointset = self.cluster(self.activepoint_set)#Cluster and find centroids
        clust_points = np.empty(self.number_of_clusters,dtype=object)
        ellipsoids = np.empty(self.number_of_clusters,dtype=object)
        print str(self.number_of_clusters)
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
                    print str(i)
                    invalid.append(i)
            else:
                ellipsoids[i] = None
                print str(i)
                invalid.append(i)
        ellipsoids = np.delete(ellipsoids, invalid)         
        return ellipsoids

    
    """This method is responsible for sampling from the enlarged ellipsoids with certain probability
    The method also checks if any ellipsoids are overlapping and behaves accordingly """
    def sample(self):
        vols = np.array([i.volume for i in self.ellipsoid_set])
        vols = vols/np.sum(vols)
        arbit = np.random.random()
        trial = Source()
        clust = Source()
        z = None
        for i in range(len(vols)):
            if(vols[i] > arbit):
                z = i
                break
        points = self.ellipsoid_set[i].sample(n_points=20)
        max_likelihood = self.LC
        count = 0
        while count<20:
            trial.X = points[count][0]
            trial.Y = points[count][1]
            trial.A = points[count][2]
            trial.R = points[count][3]
            trial.logL = log_likelihood(trial)
            self.number+=1

            if(trial.logL > max_likelihood):
                clust.__dict__ = trial.__dict__.copy()
                max_likelihood = trial.logL
                self.found == True

            count+=1
        if(self.found == True): return clust, self.number;
        else : return None, self.number;       



        


"""Class for ellipsoids"""

class Ellipsoid(object):

    def __init__(self, points, enlargement_factor):

        self.clpoints = points
        self.centroid = np.mean(points,axis=0)
        self.enlargement_factor = enlargement_factor
        self.covariance_matrix = self.build_cov(self.centroid, self.clpoints)
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
        dim = 4
        points = np.empty((n_points, dim), dtype = float)
        values, vects = np.linalg.eig(self.covariance_matrix)
        scaled = np.dot(vects, np.diag(np.sqrt(values)))
        for i in range(n_points):
            randpt = np.random.randn(dim)
            point  = randpt* np.random.rand()**(1./dim) / np.sqrt(np.sum(randpt**2))
            points[i, :] = np.dot(scaled, point) + self.centroid
        return points

    def find_volume(self):
        return (np.pi**2)*(np.sqrt(np.linalg.det(self.covariance_matrix)))/2.

        
        


if __name__ == '__main__':
    sources = get_sources(300)
    clustellip = Clustered_Sampler(active_samples = sources, enlargement= 1.0)
    X0 = [i.X for i in sources]
    Y0 = [i.Y for i in sources]
    ellipsoids = clustellip.optimal_ellipsoids()
    plt.figure()
    ax = plt.gca()
    points = []
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
    plt.plot(X, Y, 'bo')
    plt.plot(X0,Y0, 'ro')
    plt.show()
     


