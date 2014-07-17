"""A program to visualize samples from NS using a base likelihood level"""

import pickle
import numpy as np
from plot import *
from sources import *

file = "sub_40000_4000"

f = open(file,'r')

data = pickle.load(f)
f.close()




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

baselikelihood = without_duplicates(data)
print str(baselikelihood)
start = 25000
stop = 30000
spec = sorted(data, key=lambda x: x.logL,reverse=True)
sortspec = [data[i] for i in range(start, stop)]
#sortspec = [data[i] for i in range(start,stop) if data[i].logL > baselikelihood]
specX = [i.X for i in sortspec]
specY = [100-i.Y for i in sortspec]
#specA = [i.A for i in sortspec]
#a = set(sortspec)
print len(sortspec)
plot_histogram(specX, bins=400)
plot_histogram(specY, bins =100)
#plot_histogram(specA, bins =1000)        
show_scatterplot(specX, specY)