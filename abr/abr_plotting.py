import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rint(arr):
    """Round with rint and cast to int

    If `arr` contains NaN, casting it to int causes a spuriously negative
    number, because NaN cannot be an int. In this case we raise ValueError.
    """
    if np.any(np.isnan(np.asarray(arr))):
        raise ValueError("cannot convert arrays containing NaN to int")
    return np.rint(arr).astype(int)
def generate_colorbar(n_colors, mapname='jet', rounding=100, start=0.3, stop=1.):
    """Generate N evenly spaced colors from start to stop in map"""
    color_idxs = rint(rounding * np.linspace(start, stop, n_colors))[::-1]
    colors = plt.get_cmap(mapname, rounding)(color_idxs)
    return colors
