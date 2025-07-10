import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas


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

def plot_single_ax_abr(abr_subdf, ax, sampling_rate=16000):
    """
    PARAMETERS:
        abr_subdf: a subdf where the index is sound levels and the columns are voltages
        ax: the axis to plot it on
        t: the x axis in ms
    RETURNS:
        ax: the axis object with the plot made
    """
    t = abr_subdf.columns/sampling_rate*1000
    for label_i in abr_subdf.index.sort_values(ascending=False):
        aut_colorbar = generate_colorbar(
            len(abr_subdf.index), mapname='inferno_r', start=0.15, stop=1)
        color_df = pandas.DataFrame(aut_colorbar,
                                    index=abr_subdf.index.sort_values(ascending=True))
        ax.plot(t, abr_subdf.loc[label_i].T * 1e6, lw=.75,
                color=color_df.loc[label_i], label=label_i)
    return ax