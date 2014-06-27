"""A program to visualize samples from NS using a base likelihood level"""

import pickle
import numpy as np
from plot import *

file = "sub_60000_1200_10"

f = open(file,'r')

data = pickle.load(f)
f.close()

start = 30000
stop = 59999

#spec = sorted(data, key=lambda x: x.logL,reverse=True)
sortspec = [data[i] for i in range(start, stop)]
specX = [i.X for i in sortspec]
specY = [100-i.Y for i in sortspec]
specA = [i.A for i in sortspec]
plot_histogram(specX, bins=400)
plot_histogram(specY, bins =100)
plot_histogram(specA, bins =1000)        
show_scatterplot(specX, specY)