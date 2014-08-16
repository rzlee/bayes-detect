# Bayesian Source Detection and Modeling

Bayesian Source detection and characterization in astronomical images.

This is a library for performing source detection in single band sky images using Bayesian approach. This employs Multimodal modal nested sampling procedure to detect and characterize wide range of sources present in an image.


## Authors
Krishna Chaitanya Chavati  
Edward Kim  
Robert J. Brunner

## Software Required

The whole project is written in python 2.7

###Python 2.7

Installation Instructions

####Windows
There are python installers in the following link. Choose the python 2.7 or higher installer as per your configuration

    https://www.python.org/download/
and also Christoph Gohlke provides pre-built Windows installers for many Python packages, including all of the core SciPy stack.

    http://www.lfd.uci.edu/~gohlke/pythonlibs/

####Linux

Installing python 2.7 on linux can be done in many ways. We can download the zip files from

     http://python.org/download/ 
 
I found the following link helpful.

    http://askubuntu.com/questions/101591/how-do-i-install-python-2-7-2-on-ubuntu


------

This package makes use of some numerical and graphical plotting packages like numpy, matplotlib, scipy, astropy etc..

###Numpy, Scipy, ipython, matplotlib, ipython-notebook 

Installation instructions

####Windows
Find the packages in this link for windows installers.

    http://www.lfd.uci.edu/~gohlke/pythonlibs/


####Linux

Linux can install the above packages easily by running the following command in the terminal

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook


###Astropy

Astropy is a core package for astronomy.

Installation instructions

####Windows
Find the package in this link for windows installers.

    http://www.lfd.uci.edu/~gohlke/pythonlibs/


####Linux
Using pip

To install Astropy with pip, simply run:

    pip install --no-deps astropy
    
To build from source. Visit the following link and follow the instructions     

    https://github.com/astropy/astropy


## Running the Program

The program can be run in two ways. One is "Manual" mode and another is "ipython" mode

###Manual

Here we can manually set the configuration to use in the sampling procedure. There is a file named "config.cfg" in "Src" folder which contains the values on which the program depends. The contents are shown here

    #Configuration file for the sampler. Contains image path , prior bounds, Type of sampler, Number of active points etc..,

    # Path of the Image to be used
    IMAGE_PATH=C:/Users/chaithuzz2/Desktop/Bayes_detect/assets/simulated_images/multinest_toy_noised

    OUTPUT_DATA_PATH=C:/Users/chaithuzz2/Desktop/Bayes_detect/output/samples_DB_test_6.dat

    # Prior bounds
    X_PRIOR_UPPER=200.0
    X_PRIOR_LOWER=0.0

    Y_PRIOR_UPPER=200.0
    Y_PRIOR_LOWER=0.0

    A_PRIOR_UPPER=12.0
    A_PRIOR_LOWER=1.0

    R_PRIOR_UPPER=9.0
    R_PRIOR_LOWER=2.0

    STOP_BY_EVIDENCE=0  

    #noise rms
    NOISE=1.0 

    # Sampler type:  "metropolis" or "clustered_ellipsoidal" or "uniform"
    SAMPLER=clustered_ellipsoidal

    # Number of active points for the nested sampler method
    ACTIVE_POINTS=1200

    # Maximum number of iterations
    MAX_ITER=13000

    # Dispersion to use in metropolis
    DISPERSION=8.0

set the values as per the need without adding extra space.     

----
After setting up the config file. There is a .py file main.py which is repsonsible for making the source and running the source detection. Run the following command 
    
    python main.py

###Ipython

We can also see the working of the program in ipython notebook. Run the following command to open it and explore.

    ipython notebook Bayesian_Source_Detection.ipynb
    
##Output
The program outputs posterior samples with relevant information. These are stored in the folder named "output"

##References
[1] Multinest paper by Feroz and Hobson et al(2008)

[2] Data Analysis: A Bayesian Tutorial by Sivia et al(2006)

[3] http://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm

[4] http://www.inference.phy.cam.ac.uk/bayesys/

[5] Hobson, Machlachlan, Bayesian object detection in astronomical images(2003)

##Notes
Work in progress

##Contact
Krishna Chaitanya Chavati

Email: chaithukrishnazz22gmail.com