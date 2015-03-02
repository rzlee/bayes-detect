from numpy import *
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from ConfigParser import SafeConfigParser

from scipy.signal import argrelextrema

import seaborn as sns #makes the plots look pretty

parser = SafeConfigParser()
parser.read("../nested_som/config.ini")

width = int(parser.get("Image", "width"))
height = int(parser.get("Image", "height"))

amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))
x,y,r,a,L=loadtxt('../nested_som/out_points_som.txt',unpack=True)


def smooth(values, window_len = 7, window="flat"):
    #window_len must be odd
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'" 

    s = r_[values[window_len-1:0:-1],values,values[-1:-window_len:-1]]
    if window == "flat":
        w=ones(window_len,'d')
    else:
        w=eval(window+'(window_len)')
    result=convolve(w/w.sum(),s,mode='valid')
    return result[:values.shape[0]]

def binned_max(xvalues, yvalues, start, stop, num_points):
    points=linspace(start,stop,num_points+1)
    bins=0.5*(points[1:]+points[:-1])
    Lval=zeros(len(bins))
    bin_width=points[1]-points[0] #compute size of each bin
    idx=floor(xvalues/bin_width) #reindex into bins
    idx=idx.astype('int')
    for i in xrange(len(bins)):
        wi=where(idx == i)[0] #getting points inside the bin
        if shape(wi)[0] == 0: continue
        #if theres nothing there just continue
        else:
            Lval[i]=max(yvalues[wi]) #bin's value is the biggest value in the bin

    mask= (Lval != 0.)
    w=where(yvalues > min(Lval))[0]
    return (w, mask, bins, Lval)

def compute_mins(xlocs, yvals, window_size = 10):
    yval_locs = argrelextrema(yvals, less, order = window_size)[0]
    return xlocs[yval_locs]

def compute_maxes(xlocs, yvals, window_size = 10):
    yval_locs = argrelextrema(yvals, greater, order = window_size)[0]
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
        
    return array(points)

def random_color():
    return plt.cm.gist_ncar(random.random())

def plot_segments(ax, locs, vals, min_vals, max_vals):
    """
    plots each segment with a different color
    where a segment should contain one peak
    """
    intervals = compute_intervals(min_vals, max_vals)
    intervals = floor(intervals).astype("int")
    for x,y in intervals:
        lower_mask = locs > x
        upper_mask = locs < y
        mask = logical_and(lower_mask, upper_mask)
        ax.plot(locs[mask], vals[mask], color=random_color())
        #color is chosen randomly, so sometimes it makes a bad selection

#first plot of parameter vs L
fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(2,3,2)

ax1.scatter(x,y,s=3,marker='.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('all posteriors before cut')
ax1.set_xlim(0,width)
ax1.set_ylim(0,height)

w, xmask, xm, Lmx = binned_max(x, L, 0, width, 350)

ax2=fig.add_subplot(2,3,1)
ax2.plot(x[w],L[w],'k,')
ax2.set_xlabel('X')
ax2.set_ylabel('Likelihood')
ax2.plot(xm[xmask],Lmx[xmask],'r-')
smoothed_x = smooth(Lmx[xmask])
ax2.plot(xm[xmask], smoothed_x, 'g-')
mins = compute_mins(xm[xmask], smoothed_x)
maxes = compute_maxes(xm[xmask], smoothed_x)
"""
#plots vertical lines
[ax2.axvline(x = val, c="b") for val in mins]
[ax2.axvline(x = val, c="r") for val in maxes]
"""
plot_segments(ax2, xm[xmask], smoothed_x, mins, maxes)
ax2.set_title('X vs Likelhood after cut')


ax3=fig.add_subplot(2,3,3)

ax3.scatter(x[w],y[w],s=3,marker='.')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_xlim(0,width)
ax3.set_ylim(0,height)
ax3.set_title('posteriors after cut')

w, ymask, ym, Lmy = binned_max(y, L, 0, height, 350)

ax4=fig.add_subplot(2,3,4)
ax4.plot(y[w],L[w],'k,')
ax4.set_xlim(0, width)
ax4.set_xlabel('Y')
ax4.set_ylabel('Likelihood')
ax4.plot(ym[ymask],Lmy[ymask],'r-')
smoothed_y = smooth(Lmy[ymask])
ax4.plot(ym[ymask], smoothed_y, 'g-')

mins = compute_mins(ym[ymask], smoothed_y)
maxes = compute_maxes(ym[ymask], smoothed_y)
"""
[ax4.axvline(x = val, c="b") for val in mins]
[ax4.axvline(x = val, c="r") for val in maxes]
"""
plot_segments(ax4, ym[ymask], smoothed_y, mins, maxes)

ax4.set_title('Y vs Likelhood after cut')

w, rmask, rm, Lmr = binned_max(r, L, rad_min, rad_max, 350)
    
ax5=fig.add_subplot(2,3,5)
ax5.plot(r[w],L[w],'k,')
ax5.set_xlim(rad_min, rad_max)
ax5.set_xlabel('R')
ax5.set_ylabel('Likelihood')
ax5.plot(rm[rmask],Lmr[rmask],'r-')
smooth_r = smooth(Lmr[rmask])
ax5.plot(rm[rmask], smooth_r, 'g-')
ax5.set_title('R vs Likelhood after cut')


w, amask, am, Lma = binned_max(a, L, amp_min, amp_max, 350)

ax6=fig.add_subplot(2,3,6)
ax6.plot(a[w],L[w],'k,')
ax6.set_xlim(amp_min, amp_max)
ax6.set_xlabel('A')
ax6.set_ylabel('Likelihood')
ax6.plot(am[amask],Lma[amask],'r-')
smooth_a = smooth(Lma[amask])
ax6.plot(am[amask], smooth_a, 'g-')
ax6.set_title('A vs Likelhood after cut')

plt.savefig('plots/summary.png',bbox_inches='tight')

"""
#second plot of 3d parameters (x,y) vs L
fig= plt.figure()

proj = fig.add_subplot(111, projection='3d')
proj.scatter(x[w],y[w],L[w],s=3,marker='.')
proj.set_xlim(0,width)
proj.set_ylim(0,height)
proj.set_xlabel('X')
proj.set_ylabel('Y')
proj.set_zlabel('Likelihood')
proj.set_title('Posteriors in 3D after cut')
"""

plt.show()


"""

#DBSCAN

XX=zeros((len(w),4))
XX[:,0]=x[w]
XX[:,1]=y[w]
#XX[:,2]=r[w]
#XX[:,3]=a[w]
#XX[:,2]=L[w]

XX = StandardScaler().fit_transform(XX)

db = DBSCAN(eps=0.07, min_samples=15).fit(XX)

core_samples_mask = zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


print n_clusters_ , 'Clusters'

unique_labels = set(labels)
colors = plt.cm.jet(linspace(0, 1, len(unique_labels)))

plt.figure()


for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        continue

    class_member_mask = (labels == k)

    xy = XX[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=3)

    xy = XX[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=3)







"""
