import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import *
from sources import *


"""Method to show the sources at the end of the nested sampling method"""
def show_source(height, width, sources):
	x = np.arange(0, width)
	y = np.arange(0, height)
	xx, yy = np.meshgrid(x, y, sparse=True)
	z = np.zeros((height,width),float)
	for i in sources:
		z += i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
	plt.imshow(z)
	plt.show()

"""Method to show the samples at the end of the nested sampling method"""
def show_samples(height, width, samples):
    min_likelihood = 999999999
    for i in samples:
    	if i.logL < min_likelihood:
    		min_likelihood = i.logL
	arr = np.zeros((height,width),float)
	for i in samples:
		arr[int(i.Y)][int(i.X)] = i.logL + abs(min_likelihood)
	plt.imshow(arr)
	plt.show()


def plot_histogram(data, bins):
	plt.hist(data,bins)
	plt.show()
	return None

# FIX ME : Figure out a way to embed information and
# show on the plot to scale and also shouls be immune to interactive transformations
def show_scatterplot(X,Y):
	plt.scatter(X,Y, marker ="*")
	plt.show()


"""Method to show the sources at the end of the nested sampling method"""
def show_source():
	height =100
	width = 400
	i = Source()
	i.X = 200
	i.Y = 50
	i.A = 100
	i.R = 3
	x = np.arange(0, width)
	y = np.arange(0, height)
	xx, yy = np.meshgrid(x, y, sparse=True)
	z = np.zeros((height,width),float)
	z = i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
	plt.imshow(z)
	plt.show()

