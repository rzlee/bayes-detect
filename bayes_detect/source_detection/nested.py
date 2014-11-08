import numpy as np

from math import exp, log
from sampler import Sampler
from mh import Metropolis_Sampler
from clustered import Clustered_Sampler
from uniform import Uniform_Sampler
from source import Source


class Nested_Sampler(Sampler):

    """
    An Implementation of Nested Sampling proposed by John Skilling and Sivia.


    Attributes
    ----------
    no_active_samples : int
        number of active samples
    maximum_iterations : int
        maximum number of iterations
    sample : str
        sampling type
    convergence_threshold : float
        stopping criterion based on evidence
    active_samples : array
        array containing the active samples
    log_evidence : float
        Log evidence
    posterior_inferences : array
        Posterior samples
    log_width : float
        Log width of the prior
    Information : float
        Information for error estimation in evidence
    no_likelihood : int
        To keep track of number of likelihood evaluations made


    References
    ----------
    .. [1] http://www.inference.phy.cam.ac.uk/bayesys/
    .. [2] Sivia D., Skilling J., 2006, Data Analysis; a Bayesian tutorial,
           2nd ed., Oxford University Press, Oxford
    .. [3] Shaw, Bridges, Hobson 2007, MNRAS, 378, 1365

    This method is based on Nested sampling(2004) method proposed by John Skilling and Sivia.

    """

    def __init__(self, data_map, params, sampler = "metropolis", conv_thresh=0.1):

        """
        Initializes the nested sampler.

        Parameters
        ----------
        no_active_samples : int
            Number of active points which aid in sampling
        max_iter : int
            Maximum number of iterations to run
        sample : str
            Sampling mode

            * "uniform" = Samples the points randomly from a uniform distribution.
            * "metropolis" = Samples the points according to Metropolis principle.
            * "clustered_ellipsoidal" = Samples the points according to Clustered ellipsoidal method.

        conv_thresh : float
            Stopping criterion based on the current evidence in an iteration.

        """

        if conv_thresh <= 0:
            ValueError("Convergence threshold must be non negative")

        Sampler.__init__(self, data_map, params)

        self.no_active_samples     = params['n']
        self.maximum_iterations    = params['max_iter']
        self.params                = params
        self.sampler_type           = sampler
        self.convergence_threshold = conv_thresh
        self.active_samples        = self.get_sources(self.no_active_samples)
        self.log_evidence          = None # Log evidence
        self.posterior_inferences  = []   # Posterior samples
        self.log_width             = None # Log width of the prior
        self.Information           = None # Information for error estimation in evidence
        self.no_likelihood         = self.no_active_samples
        # To keep track of number of likelihood evaluations made
        self.ellipsoids            = None
        self.sampler               = None
        self.params                = params


    def fit(self):

        """
        Runs the nested sampling procedure.

        Returns
        -------
        A dict mapping the following to their values.

            *  src - Active points
            *  samples - Posterior samples
            *  logZ - The log evidence
            *  Information - The Information for error estimation
            *  likelihood_calculations - Number of likelihood evaluations
            *  iterations - Number of iterations until stopping

        """
        #self note: we implementation of clustered_sampler will be a special case of "new"

        #Initializing evidence and prior mass
        self.log_evidence = -1e300
        self.log_width = log(1.0 - exp(-1.0 / self.no_active_samples))
        self.Information = 0.0
        iteration = 1

        self.sampler = self.setup_sampler(self.data_map, self.params, self.active_samples)

        while True: #we run until either max_iter or convergence
            smallest = 0

            self.active_samples = filter(lambda x: x.logL != None, self.active_samples)
            #clean out invalid values
            LogL = [i.logL for i in self.active_samples]
            #Finding the object with smallest likelihood
            smallest = np.argmin(LogL)

            #Assigning local evidence to the smallest sample
            self.active_samples[smallest].logWt = self.log_width + self.active_samples[smallest].logL;

            largest = np.argmax(LogL)


            #Calculating the updated evidence
            temp_evidence = np.logaddexp(self.log_evidence, self.active_samples[smallest].logWt)

            #Calculating the information which will be helpful in calculating the uncertainity
            self.Information = exp(self.active_samples[smallest].logWt - temp_evidence) * self.active_samples[smallest].logL + \
            exp(self.log_evidence - temp_evidence) * (self.Information + self.log_evidence) - temp_evidence;

            self.log_evidence = temp_evidence

            #compute value that may be used for the stopping criterion
            stopping = self.active_samples[largest].logL + self.log_width - self.log_evidence

            if iteration%1000 == 0 or iteration==1:
                print "Iteration: "+str(iteration) + "  maxZ: "+str(stopping)

            if stopping < self.convergence_threshold and self.params['stop_by_evidence']==True:
                break

            if iteration >= self.maximum_iterations and self.params['stop_by_evidence']==False:
                break

            sample = Source()
            sample.__dict__ = self.active_samples[smallest].__dict__.copy()

            #storing posterior points
            self.posterior_inferences.append(sample)

            #New likelihood constraint
            likelihood_constraint = self.active_samples[smallest].logL

            updated, number = self.draw_sample(self.active_samples, iteration)
            self.active_samples[smallest].__dict__ = updated.__dict__.copy()
            LogL[smallest] = self.active_samples[smallest].logL
            self.no_likelihood = number

            #Shrink width
            self.log_width -= 1.0 / self.no_active_samples;
            iteration += 1  #increment iteration

        # FIX ME: Incorporate the active samples into evidence calculation and information after the loop
        return { "src":self.active_samples,
            "samples":self.posterior_inferences,
            "logZ":self.log_evidence,
            "Information":self.Information,
            "likelihood_calculations":self.no_likelihood,
            "iterations":iteration
            }

    def setup_sampler(self, data_map, params, active_samples):
        #we do the setup of the various samplers in here
        #first we need to know what sampler we are going to be dealing with
        sampler_type = params['type']
        smallest = int(np.argmin([i.logL for i in active_samples]))
        like_constraint = self.active_samples[smallest].logL
        like_calc = self.no_likelihood # number of active samples

        sampler = None

        if sampler_type == "uniform":
            sampler = Uniform_Sampler(data_map, params,
                                      likelihood_constraint = like_constraint,
                                      no = like_calc)

        elif sampler_type == "metropolis":
            while True:
                survivor = int(self.no_active_samples * np.random.uniform(0,1)) % self.no_active_samples  # force 0 <= copy < n
                if survivor != smallest:
                    break
            obj = active_samples[survivor]
            sampler = Metropolis_Sampler(data_map, params, to_evolve = obj,
                                         likelihood_constraint = like_constraint,
                                         no = like_calc)

        elif sampler_type == "clustered_sampler":
            sampler = Clustered_Sampler(data_map, params, active_samples = active_samples,
                                        likelihood_constraint = like_constraint, enlargement = 1.0,
                                        no = like_calc)
            self.wait = params['wait']
        else:
            raise Exception("invalid sampler requested")

        return sampler

    def draw_sample(self, active_samples, num_iter):
        #wrap interaction with samplers inside here
        if self.sampler_type == "uniform":
            return self.sampler.sample()
        if self.sampler_type == "metropolis":
            res = self.sampler.sample()
            self.sampler = self.setup_sampler(self.data_map, self.params, active_samples)
            return res
        if self.sampler_type == "clustered_sampler": 
            if self.wait == 0 or num_iter % self.wait == 0:
                self.sampler = self.setup_sampler(self.data_map, self.params, active_samples)
            return self.sampler.sample()
