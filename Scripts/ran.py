"""A program to visualize samples from NS using a base likelihood level"""

#import pickle
import numpy as np
#from plot import *
#from sources import *
import sys
from sklearn.cluster import DBSCAN

#file = "posterior"

#f = open(file,'r')

#data = pickle.load(f)
#f.close()


def without_duplicates(data):
    logl = set()
    original = []
    l = []
    for i in range(len(data)):
        if(data[i].logL not in logl):
            src = Source()
            src.__dict__ = data[i].__dict__.copy()
            logLike = src.logL 
            original.append(logLike)
            logl.add(logLike)
    print str(len(original))        
    mean = np.mean(original)
    std = np.std(original)
    
    return 10*mean    

#baselikelihood = without_duplicates(data)
#print str(baselikelihood)
#start = 25000
#stop = 30000
#spec = sorted(data, key=lambda x: x.logL+x.logWt)#,reverse=True)
#for i in range(8):
#    print str(spec[i].X)+" "+str(spec[i].Y)+" "+str(spec[i].A)+" "+str(spec[i].R)+" "+str(spec[i].logL)+" "+str(spec[i].logL+spec[i].logWt)
#sortspec = [data[i] for i in range(start, stop)]
#sortspec = [data[i] for i in range(start,stop) if data[i].logL > baselikelihood]
#specX = [i.X for i in sortspec]
#specY = [100-i.Y for i in sortspec]
#specA = [i.A for i in sortspec]
#a = set(sortspec)
#print len(sortspec)
#plot_histogram(specX, bins=400)
#plot_histogram(specY, bins =100)
#plot_histogram(specA, bins =1000)        
#show_scatterplot(specX, specY)

File = open('C:/Users/chaithuzz2/Desktop/Bayes_detect/output/samples_DB_test_5.dat','r')
samples = []
Lines = File.readlines()

for line in Lines:
    if line[0]=='\n':
        pass
    else:
        a = line.split(" ")
        samples.append([float(a[0]), float(a[1])])

db = DBSCAN(eps=3, min_samples=10).fit(samples)
labels = db.labels_
number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#print "number "+str(number_of_clusters)
#for i in range(number_of_clusters):

#    b = [samples[k] for k in range(len(samples)) if labels[k] == i]
#    print np.mean(b, axis=0) 


