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
        self.sample                = sampler
        self.convergence_threshold = conv_thresh
        self.active_samples        = self.get_sources(self.no_active_samples)
        self.log_evidence          = None # Log evidence
        self.posterior_inferences  = []   # Posterior samples
        self.log_width             = None # Log width of the prior
        self.Information           = None # Information for error estimation in evidence
        self.no_likelihood         = self.no_active_samples
        # To keep track of number of likelihood evaluations made
        self.ellipsoids            = None
        
    


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

        #Initializing evidence and prior mass
        self.log_evidence = -1e300
        self.log_width = log(1.0 - exp(-1.0 / self.no_active_samples))
        self.Information = 0.0
        LogL = [i.logL for i in self.active_samples]
        iteration = None
        stop = None
        prev_stop = 0.0

        for iteration in range(1,60000):
            smallest = 0

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

            # FIX ME : Add a stopping criterion condition

            self.log_evidence = temp_evidence

            stopping = self.active_samples[largest].logL + self.log_width - self.log_evidence


            if iteration%1000 == 0 or iteration==1:
                print "Iteration: "+str(iteration) + "  maxZ: "+str(stopping)

            if stopping < self.convergence_threshold and self.params['stop_by_evidence']==1:
                break


            if iteration >= self.maximum_iterations and self.params['stop_by_evidence']==0:
                break

            sample = Source()
            sample.__dict__ = self.active_samples[smallest].__dict__.copy()

            #storing posterior points
            self.posterior_inferences.append(sample)

            #New likelihood constraint
            likelihood_constraint = self.active_samples[smallest].logL

            survivor = int(smallest)

            while True:
                survivor = int(self.no_active_samples * np.random.uniform(0,1)) % self.no_active_samples  # force 0 <= copy < n
                if survivor != smallest:
                    break

            if self.sample == "metropolis":
                #Obtain new sample using Metropolis principle
                updated, number = self.metropolis_sampling(obj = self.active_samples[survivor],
                                                           LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number

            if self.sample == "clustered_ellipsoidal":
                #Obtain new sample using Clustered ellipsoidal sampling
                updated, number = self.clustered_sampling(active_points = self.active_samples,
                                                          LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number

            if self.sample == "uniform":
                #Obtain new sample using uniform sampling principle
                updated, number = self.uniform_sampling(LC = likelihood_constraint, likelihood_calc =self.no_likelihood)
                self.active_samples[smallest].__dict__ = updated.__dict__.copy()
                LogL[smallest] = self.active_samples[smallest].logL
                self.no_likelihood = number

            if self.sample == "new":

                if iteration ==1 or iteration%30==0 :
                    Clust_ellip = Clustered_Sampler(self.data_map, self.params, active_samples=self.active_samples,
                                                    likelihood_constraint= likelihood_constraint,
                                                    enlargement=1.0, no=self.no_likelihood)
                    self.ellipsoids = Clust_ellip.ellipsoid_set
                found = 0
                r_l, r_u = self.getPrior_R()
                a_l, a_u = self.getPrior_A()
                while found == 0:
                    arbit = np.random.uniform(0,1)
                    trial = Source()
                    clust = Source()
                    z = int((len(self.ellipsoids))*arbit)
                    points = None
                    try:
                        points = self.ellipsoids[z].sample(n_points=50)
                    except IndexError:
                        print "\n"
                        print "\n"
                        print "Please adjust the clustering parameters and try again."
                        print "\n"
                        print "\n"
                    max_likelihood = likelihood_constraint
                    count = 0
                    while count<50:
                        trial.X = points[count][0]
                        trial.Y = points[count][1]
                        trial.A = np.random.uniform(a_l,a_u)
                        trial.R = np.random.uniform(r_l,r_u)
                        trial.logL = self.log_likelihood(trial)
                        self.no_likelihood+=1

                        if(trial.logL > max_likelihood):
                            clust.__dict__ = trial.__dict__.copy()
                            max_likelihood = trial.logL
                            found = 1
                            break

                        count+=1
                    self.active_samples[smallest].__dict__ = clust.__dict__.copy()
                    LogL[smallest] = self.active_samples[smallest].logL

            #Shrink width
            self.log_width -= 1.0 / self.no_active_samples;

        # FIX ME: Incorporate the active samples into evidence calculation and information after the loop
        return { "src":self.active_samples,
            "samples":self.posterior_inferences,
            "logZ":self.log_evidence,
            "Information":self.Information,
            "likelihood_calculations":self.no_likelihood,
            "iterations":self.maximum_iterations
            }


    def metropolis_sampling(self, obj, LC, likelihood_calc):

        """
        Returns the sample satisfying the likelihood condition by metropolis sampling

        Parameters
        ----------
        obj : object
            The sample to evolve
        LC  : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        evolved - object
            The evolved sample satisfying the likelihood constraint
        number - int
            The updated likelihood calculations number

        """

        #Instantiating the metropolis sampler object
        Metro = Metropolis_Sampler(self.data_map, self.params, to_evolve = obj, likelihood_constraint = LC, no =likelihood_calc )
        evolved, number = Metro.sample()
        return evolved, number


    def clustered_sampling(self, active_points, LC, likelihood_calc ):

        """
        Returns the sample satisfying the likelihood condition by clustered ellipsoidal sampling

        Parameters
        ----------
        active_points : array
            The full set of active points at current state
        LC : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        sample : object
            The evolved sample satisfying the likelihood constraint
        number : int
            The updated likelihood calculations number

        """


        Clust = Clustered_Sampler(self.data_map, self.params, active_samples=active_points,
                                  likelihood_constraint=LC, enlargement=1.0, no=likelihood_calc)
        sample = None
        number = None
        while True:
            sample, number = Clust.sample()
            if(sample.logL > LC):
                break
            Clust = Clustered_Sampler(self.data_map, self.params,
                                      active_samples=active_points, likelihood_constraint=LC,
                                      enlargement=1.0, no=number)
        return sample, number


    def uniform_sampling(self, LC, likelihood_calc):

        """
        Returns the sample satisfying the likelihood condition by uniform random sampling

        Parameters
        ----------
        LC  : float
            likelihood constraint
        likelihood_calc : int
            Number of likelihood calculations until this point

        Returns
        -------
        evolved : object
            The evolved sample satisfying the likelihood constraint
        number : int
            The updated likelihood calculations number

        """

        unif = Uniform_Sampler(self.data_map, self.params, likelihood_constraint = LC, no =likelihood_calc)
        evolved, number = unif.sample()
        return evolved, number
