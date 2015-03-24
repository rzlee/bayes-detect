"""
.. module:: model_nest_som
.. moduleauthor:: Matias Carrasco Kind

"""
__author__ = 'Matias Carrasco Kind'

"""
Many things to do, and comment and fix
This is preliminary
"""
from numpy import *
import matplotlib.pyplot as plt
#import pyfits as pf
import sys,os
import scipy
import timeit
from scipy import stats
import SOMZ as som
import pickle
import matplotlib.cm as cm
from scipy.spatial import Voronoi,voronoi_plot_2d
import copy
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg

from ConfigParser import SafeConfigParser


def voronoi_plot_2d_local(vor, ax=None):
    ver_all=vor.vertices
    ncurr=len(vor.vertices)
    ntot=len(vor.vertices)
    nfix=len(vor.vertices)
    for simplex in vor.ridge_vertices:
        simplex = asarray(simplex)
        if all(simplex >= 0):
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-', lw=2)
    ptp_bound = vor.points.ptp(axis=0)
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = asarray(simplex)
        if any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= linalg.norm(t)
            n = array([-t[1], t[0]])  # normal
            midpoint = vor.points[pointidx].mean(axis=0)
            direction = sign(dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()
            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k-',lw=2)
            ver_all=concatenate((ver_all,[[far_point[0],far_point[1]]]))
    for s in vor.ridge_vertices:
        if -1 in s:
            s[s.index(-1)]=ncurr
            ncurr+=1
    return ver_all

def make_plot(points,AC,name):
                  
    fig=plt.figure(1,figsize=(15,10))
    ax1=fig.add_subplot(2,3,1)  
    ax1.plot(points[:name,0],points[:name,1],'k.')
    ax1.set_xlim(0,width)
    ax1.set_ylim(0,height)
    ax1.set_title('Posterior points')
    ax1.set_yticks([])
    ax1.set_xticks([])


    
    ax3=fig.add_subplot(2,3,2)
    xt=points[:name,0]
    yt=points[:name,1]
    hh,locx,locy=scipy.histogram2d(xt,yt,bins=[linspace(0,width,width+1),linspace(0,height,height+1)])
    ax3.imshow(flipud(hh.T),extent=[0,width,0,height],aspect='normal')
    ax3.set_title('Pseudo image from posterior')



    ax2=fig.add_subplot(2,3,3)
    ax2.plot(AC[:,0],AC[:,1],'k.')
    ax2.set_xlim(0,width)
    ax2.set_ylim(0,height)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('Active points')



    ax4=fig.add_subplot(2,3,4)
    ax4.imshow(flipud(data),extent=[0,width,0,height])
    ax4.set_title('Original image with noise')


    ax5=fig.add_subplot(2,3,5)
    ax5.imshow(flipud(data_or),extent=[0,width,0,height])
    ax5.set_title('Original image ')

    name="%05d" % name


    ax6=fig.add_subplot(2,3,6)
    img=mpimg.imread(output_folder + '/plots/som_'+name+'.png')
    ax6.imshow(img,extent=[0,width,0,height],aspect='normal')
    ax6.set_title('SOM map ')
    #TODO: look at this later
    os.system('rm -f ' + output_folder + '/plots/som_'+name+'.png')
    #axin.change_geometry(*(2,3,6))
    #axin= fig.axes.append(axin)

    fig.savefig(output_folder + '/plots/all6_'+name+'.png',bbox_inches='tight')
    fig.clear()
    

def make_source(src_array,height,width):
    x = arange(0, width)
    y = arange(0, height)
    xx, yy = meshgrid(x, y, sparse=True)
    z = zeros((height,width),float)
    for i in src_array:
        z+= i[2]*exp(-1*((xx-i[0])**2+(yy-i[1])**2)/(2*(i[3]**2)))
    return z

def add_gaussian_noise(mean, sd, data):
    height = len(data)
    width = len(data[0])
    my_noise=stats.distributions.norm.rvs(mean,sd,size=(height, width))
    noised = data + my_noise
    return noised

def lnlike(a,D,nlog=0):
    X=a[0]
    Y=a[1]
    A=a[2]
    R=a[3]
    noise=abs(noise_lvl)

    if X < 0 or X > width: return [-inf, nlog]
    if Y < 0 or Y > width: return [-inf, nlog]
    if A < amp_min or A > amp_max: return [-inf, nlog]
    if R < rad_min or R > rad_max: return [-inf, nlog]

    S=A*exp(-(((xx-X)**2+(yy-Y)**2))/(2.*R*R))
    DD=data-S
    DDf=DD.flatten()
    Like=-0.5*linalg.norm(DDf)**2*(1./noise) - (width * height)/2 * log(2*pi) + 4 * log(noise)
    nlog+=1
    return [Like,nlog]


def sample():
    xt=random.rand()*(width - 1.)
    yt=random.rand()*(height - 1.)
    at=random.rand()*(amp_max - amp_min) + amp_min
    rt=random.rand()*(rad_max - rad_min) + rad_min
    return array([xt,yt,at,rt])


def sample_som(jj,active,neval,LLog_min,nt=5,nit=100,create='no',sample='yes',inM=''):
    if create=='yes':
        DD=array([active[:,0],active[:,1],active[:,2],active[:,3]]).T
        lmin=min(active[:,4])
        L=active[:,4]-lmin
        lmax=max(L)
        L=L/max(L)
        M=som.SelfMap(DD,L,Ntop=nt,iterations=nit,periodic='no')
        M.create_mapF()
        M.evaluate_map()
        M.logmin=lmin
        M.logmax=lmax
        ML=zeros(nt*nt)
        for i in xrange(nt*nt):
            if M.yvals.has_key(i):
                ML[i]=mean(M.yvals[i])
        M.ML=ML
        ss=argsort(ML)
        M.ss=ss
        ML2=arange(len(ML))*1.
        ML2=ML2/sum(ML2)
        M.ML2=ML2
        if show_plot:
            #plot
            col = cm.jet(linspace(0, 1, nt*nt))
            Nr=40000
            XX=random.rand(Nr)*width
            YY=random.rand(Nr)*height
            XX=concatenate((XX,zeros(500),ones(500)*width,linspace(0,width,500),linspace(0,width,500)))
            YY=concatenate((YY,linspace(0,height,500),linspace(0,height,500),ones(500)*height,zeros(500)))
            TT=ones(len(XX))
            RR=array([XX,YY,TT,TT]).T
            M.evaluate_map(inputX=RR,inputY=zeros(len(RR)))
            figt=plt.figure(2, frameon=False)
            figt.set_size_inches(5, 5)
            ax1=plt.Axes(figt, [0., 0., 1., 1.])
            ax1.set_axis_off()
            figt.add_axes(ax1)


            for i in xrange(nt*nt):
                if M.ivals.has_key(ss[i]):
                    w=array(M.ivals[ss[i]])
                    DDD=array([XX[w],YY[w]]).T
                    if len(DDD)> 2:
                        ht=ConvexHull(DDD)
                        ax1.fill(*zip(*DDD[ht.vertices]), color=col[i], alpha=0.6)
                    #ax1.scatter(XX[w],YY[w],marker='o',edgecolor='none',color=col[i],s=50,alpha=0.2)
            cx=M.weights[0]
            cy=M.weights[1]
            ax1.plot(active[:,0],active[:,1],'k.')

            # compute Voronoi tesselation
            points2=array([cx,cy]).T
            vor = Voronoi(points2)
            pp=voronoi_plot_2d_local(vor,ax=ax1)
            ax1.set_xlim(0,width)
            ax1.set_ylim(0,height)
            #ax1.set_axis_off()
            plt.axis('off')
            plt.gca().set_axis_off()
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            
            nnn='%05d' % jj
            figt.savefig(output_folder + '/plots/som_'+nnn+'.png',bbox_inches='tight',pad_inches=0)
            figt.clear()
            M.evaluate_map(inputX=DD,inputY=L)
        
    else:
        M=inM
        ML=M.ML
        ML2=M.ML2
        ss=M.ss
    while True:
        t=random.choice(len(ML2), 1, p=ML2)[0]
        if (M.ivals.has_key(ss[t])):
            break
    cell=ss[t]

    
    while True:
        keep=True
        xt=random.normal(mean(active[M.ivals[cell],0]),max([std(active[M.ivals[cell],0]),0.01]))
        yt=random.normal(mean(active[M.ivals[cell],1]),max([std(active[M.ivals[cell],1]),0.01]))
        at=random.normal(mean(active[M.ivals[cell],2]),max([std(active[M.ivals[cell],2]),0.01]))
        rt=random.normal(mean(active[M.ivals[cell],3]),max([std(active[M.ivals[cell],3]),0.01]))
        if (xt < 0) or (xt>width) : keep=False
        if (yt < 0) or (yt>height) : keep=False
        if (at < amp_min)  or (at > amp_max) : keep=False
        if (rt < rad_min) or (rt > rad_max): keep=False

        if keep:
            new=array([xt,yt,at,rt])
            newL,neval=lnlike(new,data,nlog=neval)
            if newL > LLog_min: break
    return [M,new,neval]

parser = SafeConfigParser()
parser.read("../config.ini")

prefix = parser.get("Misc", "prefix")
location = parser.get("Misc", "location")
output_folder = location + "/" + prefix 
image_location = output_folder + "/" + prefix + "_noised.npy"
no_noise_location = output_folder + "/" + prefix + "_clean.npy"

#image parameters
width = int(parser.get("Sampling","width"))
height = int(parser.get("Sampling","height"))
image_location = parser.get("Image", "location")
no_noise_location = parser.get("Image", "no_noise_location")

#sampling parameters
noise_lvl = float(parser.get("Sampling", "noise"))
amp_min = float(parser.get("Sampling", "amp_min"))
amp_max = float(parser.get("Sampling", "amp_max"))

rad_min = float(parser.get("Sampling", "rad_min"))
rad_max = float(parser.get("Sampling", "rad_max"))

niter = int(parser.get("Sampling", "niter"))
num_active_points = int(parser.get("Sampling", "num_active"))
num_som_iter = int(parser.get("Sampling", "num_som_iter"))
num_som_points = int(parser.get("Sampling", "num_som_points"))

#output parameters
output_filename = prefix + "_" + parser.get("Output", "output_filename")

show_plot = parser.getboolean("Output", "plot")

SrcArray = [[43.71, 22.91, 10.54, 3.34],
            [101.62, 40.60, 1.37, 3.40],
            [92.63, 110.56, 1.81, 3.66],
            [183.60, 85.90, 1.23, 5.06],
            [34.12, 162.54, 1.95, 6.02],
            [153.87, 169.18, 1.06, 6.61],
            [155.54, 32.14, 1.46, 4.05],
            [130.56, 183.48, 1.63, 4.11]]

data_or=make_source(src_array = SrcArray,height=height, width=width)
noise = noise_lvl
data=add_gaussian_noise(mean=0,sd=noise,data=data_or)

#filee='../bayes-detect-master/simulated_images/multinest_toy_noised'
#s=open(filee,'r')
#data=pickle.load(s)
#s.close()

#sys.exit(0)

os.system('mkdir -p ' + output_folder + '/plots/')

data = load(image_location)
data_or = load(no_noise_location)


#Im=pf.open('ufig_20_g_sub_500_sub_small.fits')
#data=Im[0].data
#Im.close()

x=arange(width,dtype='float')
y=arange(height,dtype='float')
xx,yy=meshgrid(x,y)


start = timeit.default_timer()
neval=0
Np=num_active_points
AC=zeros((Np,5))

AC[:,0]=random.rand(Np)*(width - 1.)
AC[:,1]=random.rand(Np)*(height - 1.)
AC[:,2]=random.rand(Np)*(amp_max-amp_min) + amp_min
AC[:,3]=random.rand(Np)*(rad_max-rad_min) + rad_min

for i in xrange(Np):
    AC[i,4],neval=lnlike(AC[i,0:4],data,nlog=neval)


print 'done with active'

Niter=niter
points=zeros((Niter,5))
for i in xrange(Niter):
    if i%num_som_iter == 0:
        print i
    reject=argmin(AC[:,4])
    minL=AC[reject,4]
    if i%num_som_iter == 0:
        Map,new,neval=sample_som(i,AC,neval,minL,nt=4,nit=150,create='yes',sample='yes')
        make_plot(points,AC,i)
    else:
        Map,new,neval=sample_som(i,AC,neval,minL,nt=4,nit=150,create='no',sample='yes',inM=Map)
    #while True:
      #  new=sample()
#        newL,neval=lnlike(new,data,nlog=neval)
#        if newL > minL:
#            break
    newL,neval=lnlike(new,data,nlog=neval)
    points[i]=AC[reject]
    AC[reject,0:4]=new
    AC[reject,4]=newL
    


stop = timeit.default_timer()

with open(output_folder + "/stats.txt", "wb") as f:
    f.writelines(["seconds: %d\n"%(stop - start), "Log evaluations: %d"%neval])

print stop - start, 'seconds'
print neval, 'Log evaluations'




#savetxt('out_points_som.txt',points,fmt='%.6f')
savetxt(output_folder + "/" + output_filename, points,fmt='%.6f')


#daa=hh.T
#hdu0=pf.PrimaryHDU(daa)
#hdulist = pf.HDUList([hdu0])
#hdulist.writeto('output_nest.fits',clobber=True)
#plt.show()



    
