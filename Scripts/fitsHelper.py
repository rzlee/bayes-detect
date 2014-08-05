import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import os

if __name__ == '__main__':
    
    folder   = "simulated_images"
    fitsList = []
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.fits'):
                fitsList.append(os.path.join(root, file))
    
    for file in fitsList:
        with fits.open(file) as hdulist:
            scidata = hdulist[0].data
            np.savetxt(file+'.txt', scidata)    
            plt.imshow(scidata)
            plt.savefig(file+".jpg")
            hdulist.close()
            
    

