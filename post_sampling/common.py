import numpy as np
from scipy.signal import argrelextrema

def smooth(values, window_len = 7, window="flat"):
    #window_len must be odd
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'" 

    s = np.r_[values[window_len-1:0:-1],values,values[-1:-window_len:-1]]
    if window == "flat":
        w=np.ones(window_len,'d')
    else:
        w=np.eval(window+'(window_len)')
    result=np.convolve(w/w.sum(),s,mode='valid')
    return result[:values.shape[0]]

def binned_max(xvalues, yvalues, start, stop, num_points):
    points=np.linspace(start,stop,num_points+1)
    bins=0.5*(points[1:]+points[:-1])
    Lval=np.zeros(len(bins))
    bin_width=points[1]-points[0] #compute size of each bin
    idx=np.floor(xvalues/bin_width) #reindex into bins
    idx=idx.astype('int')
    for i in xrange(len(bins)):
        wi=np.where(idx == i)[0] #getting points inside the bin
        if np.shape(wi)[0] == 0: continue
        #if theres nothing there just continue
        else:
            Lval[i]=max(yvalues[wi]) #bin's value is the biggest value in the bin

    mask= (Lval != 0.)
    w=np.where(yvalues > min(Lval))[0]
    return (w, mask, bins, Lval)

def compute_mins(xlocs, yvals, window_size = 10):
    yval_locs = argrelextrema(yvals, np.less, order = window_size)[0]
    return xlocs[yval_locs]

def compute_maxes(xlocs, yvals, window_size = 10):
    yval_locs = argrelextrema(yvals, np.greater, order = window_size)[0]
    return xlocs[yval_locs]

def compute_intervals(mins, maxes):
    #tries to compute an array of ranges thta contain a peak
    points = []
    start = None
    stop = None
    min_index = 0
    max_index = 0
    encountered_peak = False
    
    #start and stop should have at least one peak between them
    #start and stop are the locations of a min
    
    while min_index < mins.shape[0] and max_index < maxes.shape[0]:
        if start is None:
            start = mins[min_index]
            min_index += 1
        if not encountered_peak and maxes[max_index] < start:
            #the peak we are at is behind our start, so we need to find the next one
            max_index += 1
        if not encountered_peak and maxes[max_index] > start:
            #select the first peak we encounter, then try to find the next min to close off this segment
            encountered_peak = True
            max_index += 1
        if encountered_peak and min_index < mins.shape[0]:
            stop = mins[min_index]
            points.append([start, stop])
            start = None
            stop = None
        
    return np.array(points)
