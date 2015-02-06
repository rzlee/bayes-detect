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
    ax1.set_xlim(0,200)
    ax1.set_ylim(0,200)
    ax1.set_title('Posterior points')
    ax1.set_yticks([])
    ax1.set_xticks([])


    
    ax3=fig.add_subplot(2,3,2)
    xt=points[:name,0]
    yt=points[:name,1]
    hh,locx,locy=scipy.histogram2d(xt,yt,bins=[linspace(0,200,201),linspace(0,200,201)])
    ax3.imshow(flipud(hh.T),extent=[0,200,0,200],aspect='normal')
    ax3.set_title('Seudo image from posterior')



    ax2=fig.add_subplot(2,3,3)
    ax2.plot(AC[:,0],AC[:,1],'k.')
    ax2.set_xlim(0,200)
    ax2.set_ylim(0,200)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('Active points')



    ax4=fig.add_subplot(2,3,4)
    ax4.imshow(flipud(data),extent=[0,200,0,200])
    ax4.set_title('Original image with noise')


    ax5=fig.add_subplot(2,3,5)
    ax5.imshow(flipud(data_or),extent=[0,200,0,200])
    ax5.set_title('Original image ')

    name="%05d" % name


    ax6=fig.add_subplot(2,3,6)
    img=mpimg.imread('plots/som_'+name+'.png')
    ax6.imshow(img,extent=[0,200,0,200],aspect='normal')
    ax6.set_title('SOM map ')
    os.system('rm -f plots/som_'+name+'.png')
    #axin.change_geometry(*(2,3,6))
    #axin= fig.axes.append(axin)

    fig.savefig('plots/all6_'+name+'.png',bbox_inches='tight')
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


SrcArray = [[43.71, 22.91, 10.54, 3.34],
            [101.62, 40.60, 1.37, 3.40],
            [92.63, 110.56, 1.81, 3.66],
            [183.60, 85.90, 1.23, 5.06],
            [34.12, 162.54, 1.95, 6.02],
            [153.87, 169.18, 1.06, 6.61],
            [155.54, 32.14, 1.46, 4.05],
            [130.56, 183.48, 1.63, 4.11]]



data_or=make_source(src_array = SrcArray,height=200, width=200)
noise = 3.0
data=add_gaussian_noise(mean=0,sd=noise,data=data_or)

#filee='../bayes-detect-master/simulated_images/multinest_toy_noised'
#s=open(filee,'r')
#data=pickle.load(s)
#s.close()

#sys.exit(0)

os.system('mkdir -p plots/')

data=load('../image_generator/noised_num_sources-10-width-200-height-200-a_min-1-a_max-10-rad_min-1-rad_max-10-noise-3.npy')
data_or=load('../image_generator/no_noise_num_sources-10-width-200-height-200-a_min-1-a_max-10-rad_min-1-rad_max-10-noise-3.npy')


#Im=pf.open('ufig_20_g_sub_500_sub_small.fits')
#data=Im[0].data
#Im.close()

x=arange(200,dtype='float')
y=arange(200,dtype='float')
xx,yy=meshgrid(x,y)

def lnlike(a,D,nlog=0):
    X=a[0]
    Y=a[1]
    A=a[2]
    R=a[3]
    noise=2.
    if X < 0: return -inf
    if Y < 0: return -inf
    if X > 199: return -inf
    if Y > 199: return -inf
    #if A <= 0. : return -inf
    #if A > 2.*D.max() : return -inf
    #if R< 0 : return -inf
    S=A*exp(-(((xx-X)**2+(yy-Y)**2))/(2.*R*R))
    DD=data-S
    DDf=DD.flatten()
    Like=-0.5*linalg.norm(DDf)**2*(1./noise)-(20000.*log(2*pi)+0.5*20000.*log(noise))
    nlog+=1
    return [Like,nlog]


def sample():
    xt=random.rand()*199.
    yt=random.rand()*199.
    at=random.rand()*11.5+1
    rt=random.rand()*7+2.
    return array([xt,yt,at,rt])




def sample_som(jj,active,neval,LLog_min,nt=5,nit=100,create='no',sample='yes',inM=''):
    if create=='yes':
        DD=array([active[:,0],active[:,1]]).T
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
        #plot
        col = cm.jet(linspace(0, 1, nt*nt))
        Nr=40000
        XX=random.rand(Nr)*200.
        YY=random.rand(Nr)*200
        XX=concatenate((XX,zeros(500),ones(500)*200.,linspace(0,200,500),linspace(0,200,500)))
        YY=concatenate((YY,linspace(0,200,500),linspace(0,200,500),ones(500)*200.,zeros(500)))
        RR=array([XX,YY]).T
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
        ax1.set_xlim(0,200)
        ax1.set_ylim(0,200)
        #ax1.set_axis_off()
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        
        nnn='%05d' % jj
        figt.savefig('plots/som_'+nnn+'.png',bbox_inches='tight',pad_inches=0)
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
        if (xt < 0) or (xt>199.) : keep=False
        if (yt < 0) or (yt>199.) : keep=False
        if (at < 1.) : keep=False
        if (rt < 1.) : keep=False

        if keep:
            new=array([xt,yt,at,rt])
            newL,neval=lnlike(new,data,nlog=neval)
            if newL > LLog_min: break
    return [M,new,neval]


start = timeit.default_timer()
neval=0
Np=5000
AC=zeros((Np,5))

AC[:,0]=random.rand(Np)*199.
AC[:,1]=random.rand(Np)*199.
AC[:,2]=random.rand(Np)*11.5+1.
AC[:,3]=random.rand(Np)*7.+2.

for i in xrange(Np):
    AC[i,4],neval=lnlike(AC[i,0:4],data,nlog=neval)


print 'done with active'

Niter=35001
points=zeros((Niter,5))
for i in xrange(Niter):
    if i%500 == 0:
        print i
    reject=argmin(AC[:,4])
    minL=AC[reject,4]
    if i%500 == 0:
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

print stop - start, 'seconds'
print neval, 'Log evaluations'




savetxt('out_points_som.txt',points,fmt='%.6f')



#daa=hh.T
#hdu0=pf.PrimaryHDU(daa)
#hdulist = pf.HDUList([hdu0])
#hdulist.writeto('output_nest.fits',clobber=True)
#plt.show()



    
