import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.transforms import *
import math



def show_source(height, width, sources):

    """
    Shows a 2D xy plot of the sources characterized by parameters [X, Y, A, R]

    Parameters
    ----------
    height : int
        Height of the image
    width : int
        width of the image
    sources : array
        Array of source objects to show


    Examples
    --------

    >>> import plot
    >>> from sources import Source
    >>> import numpy as np
    >>> height = 200
    >>> width = 200
    >>> Sources = []
    >>> for i in range(4):
    >>>     a = Source()
    >>>     a.X = np.random.uniform(0.0, 200.0)
    >>>     a.Y = np.random.uniform(0.0, 200.0)
    >>>     a.A = np.random.uniform(1.0, 15.0)
    >>>     a.R = np.random.uniform(2.0, 9.0)
    >>>     Sources.append(a)
    >>> plot.show_source(height, width, Sources)

    .. plot::

        import plot
        from sources import Source
        import numpy as np
        height = 200
        width = 200
        Sources = []
        for i in range(4):
            a = Source()
            a.X = np.random.uniform(0.0, 200.0)
            a.Y = np.random.uniform(0.0, 200.0)
            a.A = np.random.uniform(1.0, 15.0)
            a.R = np.random.uniform(2.0, 9.0)
            Sources.append(a)
        plot.show_source(height, width, Sources)

    """

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.zeros((height,width),float)
    for i in sources:
        z += i.A*np.exp(-1*((xx-i.X)**2+(yy-i.Y)**2)/(2*(i.R**2)))
    plt.imshow(z)
    plt.show()


def plot_histogram(data, bins, title):

    """
    Shows the histogram of a 1D data.

    Parameters
    ----------
    data : array
        data to be shown in histogram format
    bins : int
        number of bins to hold
    title : str
        title of the plot

    Examples
    --------

    >>> import plot
    >>> import numpy as np
    >>> X = np.random.randn(400)
    >>> plot.plot_histogram(X, 10, "Histogram demo")

    .. plot::

        import plot
        import numpy as np
        X = np.random.randn(400)
        plot.plot_histogram(X, 10, "Histogram demo")

    """

    plt.hist(data,bins)
    plt.title(title)
    plt.show()
    return None


def show_scatterplot(X,Y,title, height, width):

    """
    Parameters
    ----------
    X : array
        x coordinates of the data
    Y : array
        y coordinates of the above data. Same shape as above
    title : str
        title of the scatter plot
    height : int
        height limit of the scatter plot
    width : int
        width limit of the scatter plot

    Examples
    --------

    >>> import plot
    >>> import numpy as np
    >>> X = np.random.randn(400)
    >>> Y = np.random.randn(400)
    >>> plot.show_scatterplot(X, Y, "scatter plot demo", 400, 400)

    .. plot::

        import plot
        import numpy as np
        X = np.random.rand(1, 400)
        Y = np.random.rand(1, 400)
        plot.show_scatterplot(X, Y, "scatter plot demo", 1, 1)

    """

    plt.scatter(X,Y, marker =".")
    plt.title(title)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()
