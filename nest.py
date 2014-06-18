from math import *
import random
from plot import *


"""Multimodal nested sampling method for source detection in astronomical images"""

"""logarithmic addition"""
def plus(x,y):
    if x>y:
        return x+log(1+exp(y-x)) 
    else:
        return y+log(1+exp(x-y))


def nested_sampling(n, max_iterations, sample_from_prior, explore):

    Src = [] # Source array
    post_inferences = [] # posterior inferences
    logwidth = None  
    logLstar = None
    H = 0.0 # Information
    logZ = -1e300 #Evidence
    logZnew = None # Updated Evidence
    copy = None
    worst = None
    nest = None

    for i in range(n):
        Src.append(sample_from_prior())
        
    logwidth = log(1.0 - exp(-1.0 / n));
    
    a = set([0,100,200,300,400,500,600,700,800,900,999])
    for nest in range(max_iterations):
        """To check progress on command line"""
        if nest in a:
            print str(nest)
        
        """Fishing out the source with lowest likelihood"""
        worst = 0
        for i in range(1, n):
            if(Src[i].logL < Src[worst].logL):
                worst = i

        Src[worst].logWt = logwidth + Src[worst].logL;

        logZnew = plus(logZ, Src[worst].logWt)
        H = exp(Src[worst].logWt - logZnew) * Src[worst].logL + \
            exp(logZ - logZnew) * (H + logZ) - logZnew;
        logZ = logZnew;

        """posterior inferences if needed"""
        post_inferences.append(Src[worst])

        if n>1: 
            while True:
                copy = int(n * random.uniform(0,1)) % n  
                if copy != worst:
                    break

        logLstar = Src[worst].logL;
        Src[worst] = Src[copy];

        """Replacing the worst source with new source that meets the likelihood constraint"""
        updated = explore(Src[worst], logLstar);
        assert(updated != None)
        Src[worst] = updated
        
        logwidth -= 1.0 / n;

    sdev_H = H/log(2.)
    sdev_logZ = sqrt(H/n)
    return {"src":Src,
            "samples":post_inferences, 
            "num_iterations":(nest+1), 
            "logZ":logZ,
            "logZ_sdev":sdev_logZ,
            "info_nats":H,
            "info_sdev":sdev_H}





