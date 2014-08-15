"""Just a script to find noiseRMS in an Image

Prints the noiseRMS in the image using scipy.stats.mode

"""

import numpy as np
from astropy.io import fits
from scipy import stats


#Reading the Image data from fits file
fitsFile = 'C:/Users/chaithuzz2/Desktop/Bayes_detect/assets/simulated_images/ufig_20_g_gal_sub_500_sub_small.fits'

hdulist   = fits.open(fitsFile)
data_map   = (hdulist[0].data)

data_map = data_map.flatten()

print str(stats.mode(data_map))
print str(max(data_map))
print str(np.mean(data_map))