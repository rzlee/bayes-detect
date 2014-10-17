class Source:

    """
     This is a class which instantiates a source object with its attributes.

    Attributes
    ----------
    X : float
        x coordinate of the center of the object
    Y : float
        y coordinate of the center of the object
    A : float
        Amplitude of the object
    R : float
        Spatial extent of the object
    logL : float
        Log likelihood of the object
    logWt : float
        Log weight of the object"""

    def __init__(self):

        self.X = None
        self.Y = None
        self.A = None
        self.R = None
        self.logL = None
        self.logWt = None
