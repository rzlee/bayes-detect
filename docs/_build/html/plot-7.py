import plot
from sources import Source
import numpy as np
height = 200
width = 200
Sources = []
for i in range(4):
    a = Source()
    a.X = np.random.uniform(0.0, 200.0)
    a.Y = np.random.uniform(0.0, 200.0)
    a.A = np.random.uniform(1.0, 15.0)
    a.R = np.random.uniform(2.0, 9.0)
    Sources.append(a) 
plot.show_source(height, width, Sources)