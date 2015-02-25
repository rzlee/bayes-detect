from numpy import *
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from ConfigParser import SafeConfigParser
from sklearn.neighbors import KernelDensity

parser = SafeConfigParser()
parser.read("../nested_som/config.ini")

width = int(parser.get("Image", "width"))
height = int(parser.get("Image", "height"))

amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))
x,y,r,a,L=loadtxt('../nested_som/out_points_som.txt',unpack=True)


ml=mean(L)
sL=std(L)
w=where(L > ml)[0]

Nb=350


fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(2,3,2)


ax1.scatter(x,y,s=3,marker='.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('all posteriors before cut')
ax1.set_xlim(0,width)
ax1.set_ylim(0,height)


xb=linspace(0,width,Nb+1)
xm=0.5*(xb[1:]+xb[:-1])
Lmx=zeros(len(xm))
dx=xb[1]-xb[0]
idx=floor(x/dx)
idx=idx.astype('int')

for i in xrange(len(xm)):
    wi=where(idx == i)[0]
    if shape(wi)[0] == 0: continue
    else:
        Lmx[i]=max(L[wi])

mask=Lmx != 0.
w=where(L > min(Lmx))[0]
#computing the cut


ax2=fig.add_subplot(2,3,1)
ax2.plot(x[w],L[w],'k,')
ax2.set_xlabel('X')
ax2.set_ylabel('Likelihood')
#ax2.plot(xm[mask],Lmx[mask],'r-')
ax2.set_title('X vs Likelhood after cut')


ax3=fig.add_subplot(2,3,3)

X,Y = mgrid[0:width, 0:height]
positions = vstack([X.ravel(), Y.ravel()])
values = vstack([x[w], y[w]])
kernel = scipy.stats.gaussian_kde(values)
Z = reshape(kernel(positions).T, X.shape)

ax3.scatter(x[w],y[w],s=3,marker='.')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.imshow(rot90(Z), cmap=plt.cm.gist_earth_r, extent=[0, width, 0, height])
ax3.set_xlim(0,width)
ax3.set_ylim(0,height)
ax3.set_title('posteriors after cut')


yb=linspace(0,200,Nb+1)
ym=0.5*(yb[1:]+yb[:-1])
Lmy=zeros(len(ym))
dy=yb[1]-yb[0]
idy=floor(y/dy)
idy=idy.astype('int')

for i in xrange(len(ym)):
    wi=where(idy == i)[0]
    if shape(wi)[0] == 0: continue
    else:
        Lmy[i]=max(L[wi])

mask=Lmy != 0.

ax4=fig.add_subplot(2,3,4)
ax4.plot(y[w],L[w],'k,')
ax4.set_xlim(0, width)
ax4.set_xlabel('Y')
ax4.set_ylabel('Likelihood')
#ax4.plot(xm[mask],Lmx[mask],'r-')
ax4.set_title('Y vs Likelhood after cut')
    
ax5=fig.add_subplot(2,3,5)
ax5.plot(r[w],L[w],'k,')
ax5.set_xlim(rad_min, rad_max)
ax5.set_xlabel('R')
ax5.set_ylabel('Likelihood')
#ax5.plot(xm[mask],Lmx[mask],'r-')
ax5.set_title('R vs Likelhood after cut')

ax6=fig.add_subplot(2,3,6)
ax6.plot(a[w],L[w],'k,')
ax6.set_xlim(amp_min, amp_max)
ax6.set_xlabel('A')
ax6.set_ylabel('Likelihood')
#ax6.plot(xm[mask],Lmx[mask],'r-')
ax6.set_title('A vs Likelhood after cut')

plt.savefig('plots/summary.png',bbox_inches='tight')

fig= plt.figure()

proj = fig.add_subplot(111, projection='3d')
proj.scatter(x[w],y[w],L[w],s=3,marker='.')
proj.set_xlim(0,width)
proj.set_ylim(0,height)
proj.set_xlabel('X')
proj.set_ylabel('Y')
proj.set_zlabel('Likelihood')
proj.set_title('Posteriors in 3D after cut')

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
