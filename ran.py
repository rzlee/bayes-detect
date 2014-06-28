"""A program to visualize samples from NS using a base likelihood level"""

import pickle
import numpy as np
from plot import *
from sources import *

file = "pickles/sub_60000_1200_10"

f = open(file,'r')

data = pickle.load(f)
f.close()



#spec = sorted(data, key=lambda x: x.logL,reverse=True)
#sortspec = [data[i] for i in range(start, stop)]
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
    #std = np.std(original)
    
    return 2*mean#+std    

baselikelihood = without_duplicates(data)
start = 30000 
stop = 45000
sortspec = [data[i] for i in range(start,stop)]# if data[i].logL > baselikelihood]
specX = [i.X for i in sortspec]
specY = [100-i.Y for i in sortspec]
specA = [i.A for i in sortspec]
a = set(specA)
print len(a)
plot_histogram(specX, bins=400)
plot_histogram(specY, bins =100)
#plot_histogram(specA, bins =1000)        
show_scatterplot(specX, specY)