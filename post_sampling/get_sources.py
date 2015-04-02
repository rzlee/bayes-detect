import numpy as np
import scipy
import common

from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read("../config.ini")

width = int(parser.get("Sampling", "width"))
height = int(parser.get("Sampling", "height"))

amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))

prefix = parser.get("Misc", "prefix")
location = parser.get("Misc", "location")
output_folder = location + "/" + prefix 

x,y,r,a,L = np.loadtxt(output_folder + "/" + prefix + "_out_points_som.txt", unpack=True)

all_vals = np.vstack((x,y,r,a,L))

def find_sources(xlower, xupper, ylower, yupper, mask, varindex):
    vals = all_vals[varindex] #select the var index we want
    smoothed = common.smooth(vals)

