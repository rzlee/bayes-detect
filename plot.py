import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager


"""Method to show the live points at the end of the nested sampling method"""
def show_source(height, width, sources, step = 0):
	x = np.arange(0, width)
	y = np.arange(0, height)
	xx, yy = np.meshgrid(x, y, sparse=True)
	z = np.zeros((height,width),float)
	for i in sources:
		z += i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
	plt.imshow(z)
	plt.show()