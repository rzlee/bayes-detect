import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import os
from matplotlib.patches import Rectangle

"""This is a method for finding all the luminous sources in a
an astronomical image in FITS format. It is based on the two pass
4-connectivity connected component labelling algorithm. The method
returns containing source objects and their pixels. The aim of this
method is to get source positions before moving to fitting them
to probabilistic models inorder to characterize them""" 

def findSources(fitsFile, plot=False):
    
    """reading the data of the fits file"""
    hdulist   = fits.open(fitsFile)
    scidata   = hdulist[0].data
    
    
    height, width    = scidata.shape   
    
    """Object map which has the shape of image data initialized
    with zeros. It will hold the object and background
    information of the data. label "1" means an object and
    label "0" indicates background."""
    objectMap = np.zeros(scidata.shape, int)
        
    """Label map which is has the shape of image data initaialized
    with zeros. It will hold the labels of the corresponding pixels
    in the connected component labelling algorithm for object
    detection. labelcount is the variable that holds label number"""
    labelMap  = np.zeros(scidata.shape, int)
    labelCount= 0
    
    """Dictionary that stores label equivalency"""
    equival   = []
    
    """Calculating the mean ans standard deviation of the
    data"""
    Mean      = np.mean(scidata)
    Stdev     = np.std(scidata)
    
    """ Threshold formula (temporary)"""
    Threshold = Mean + Stdev
    
    """ Final object dictionary which contains object name and
    pixels belong to it """
    srcDict   = {}  
    
    """Preparing the object map"""
    for i in range(height):
        for j in range(width):
            if scidata[i][j] >= Threshold :
                objectMap[i][j] = 1
    
             
    """Labelling the first row of the image matrix"""
    for j in range(width-1):
        if objectMap[0][j+1] == objectMap[0][j]:
            labelMap[0][j+1] = labelMap[0][j]
        if objectMap[0][j+1] != objectMap[0][j]:
            labelCount = labelCount + 1
            labelMap[0][j+1] = labelCount
            
    """Labelling the first column of the image matrix"""
    for i in range(height-1):
        if objectMap[i+1][0] == objectMap[i][0]:
            labelMap[i+1][0] = labelMap[i][0]
        if objectMap[i+1][0] != objectMap[i][0]:
            labelCount = labelCount + 1
            labelMap[i+1][0] = labelCount
            
    """Labelling the rest of the rows and columns"""
    for i in range(height-1):
        for j in range(width-1):
            if objectMap[i+1][j+1] == objectMap[i+1][j] and objectMap[i+1][j+1] != objectMap[i][j+1]:
                labelMap[i+1][j+1] = labelMap[i+1][j]
            if objectMap[i+1][j+1] != objectMap[i+1][j] and objectMap[i+1][j+1] == objectMap[i][j+1]:
                labelMap[i+1][j+1] = labelMap[i][j+1]
            if objectMap[i+1][j+1] != objectMap[i+1][j] and objectMap[i+1][j+1] != objectMap[i][j+1]:
                labelCount = labelCount + 1
                labelMap[i+1][j+1] = labelCount
            if objectMap[i+1][j+1] == objectMap[i][j+1] and objectMap[i+1][j+1] == objectMap[i+1][j]:
                if labelMap[i][j+1] == labelMap[i+1][j]:
                    labelMap[i+1][j+1] = labelMap[i][j+1]
                else:
                    """This is a special case where we suddenly find
                    that some labels are equivalent to each other.
                    Thats why we note the equivalency to resolve them
                    in the second pass"""
                    labelMap[i+1][j+1] = min(labelMap[i][j+1], labelMap[i+1][j])
                    temp               = max(labelMap[i][j+1], labelMap[i+1][j])   
                    equival.append([temp, labelMap[i+1][j+1]])
                    
    
    maxlabel     = np.max(labelMap)
    parentArray  = [i for i in range(maxlabel+1)]
    
    """ A find method to recursively find the parent
    of a label in the noted equivalencies"""
    def find(t, parent):
        if parent[t] == t:
            return t
        else:
            return find(parent[t], parent)
    
    """Sorting out the equivalencies in the array"""
    for i in equival:
        k,l     = i
        kParent = find(k, parentArray)
        lParent = find(l, parentArray)
        parentArray[max(kParent, lParent)] = parentArray[min(kParent, lParent)]
    
    """Relabelling the image matrix after sorting out
    the equivalencies. Second pass """
    for i in range(height):
        for j in range(width):
            labelMap[i][j] = parentArray[labelMap[i][j]]
            
    """Set to learn unique labels of the image"""
    finalLabelSet = set()
    
    for i in range(maxlabel+1):
        finalLabelSet.add(parentArray[i])
        
    labelList = list(sorted(finalLabelSet))
    
    objectCount = 0
    """writing the objects and their belonging pixels to the
    source dictionary"""
    for i in range(height):
        for j in range(width):
            if labelMap[i][j] == 0:
                continue
            else:
                try:
                    srcDict[labelList.index(labelMap[i][j])].append((i,j))
                except KeyError:
                    srcDict[labelList.index(labelMap[i][j])] = []
                    srcDict[labelList.index(labelMap[i][j])].append((i,j))
                    
                            
    """Plottint the objects on pyplot"""                        
    if plot == True:
        plt.imshow(scidata)
        offset = 10
        for i in srcDict:
            allX = [x[1] for x in srcDict[i]]
            allY = [x[0] for x in srcDict[i]]
            minX = np.min(allX)
            maxX = np.max(allX)
            minY = np.min(allY)
            maxY = np.max(allY)
            currentAxis = plt.gca()
            currentAxis.add_patch(Rectangle((minX -offset, minY -offset), maxX-minX+2*offset,
                maxY-minY+2*offset, alpha=1, facecolor ='none', edgecolor='white'))
            plt.text((minX+maxX)/2, (minY+maxY)/2, i, fontsize=8, color='white')
        plt.show()
    
    hdulist.close()
    return srcDict
    
    
if __name__ == '__main__':
    fileRoot = "simulated_images/ufig_20_g_gal_sub_500.fits"
    sources  = findSources(fileRoot, plot=True)
                        
        
    
    
        
        
            
    
        
                    
                         
                
    
        
        
        
    
    
                
            
    
    
    
    
            
    

