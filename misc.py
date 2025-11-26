"""Catchall module for useful functions"""

import matplotlib.mlab

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
    
def psd(data, NFFT=None, Fs=None, detrend='mean', window=None, noverlap=None,
    scale_by_freq=None, **kwargs):
    """Compute power spectral density.

    A wrapper around mlab.psd with more documentation and slightly different
    defaults.

    Arguments
    ---
    data : The signal to analyze. Must be 1d
    NFFT : defaults to 256 in mlab.psd
    Fs : defaults to 2 in mlab.psd
    detrend : default is 'mean', overriding default in mlab.psd
    window : defaults to Hanning in mlab.psd
    noverlap : defaults to 0 in mlab.psd
        50% or 75% of NFFT is a good choice in data-limited situations
    scale_by_freq : defaults to True in mlab.psd
    **kwargs : passed to mlab.psd

    Notes on scale_by_freq
    ---
    Using scale_by_freq = False makes the sum of the PSD independent of NFFT
    Using scale_by_freq = True makes the values of the PSD comparable for
    different NFFT
    In both cases, the result is independent of the length of the data
    With scale_by_freq = False, ppxx.sum() is roughly comparable to
      the mean of the data squared (but about half as much, for some reason)
    With scale_by_freq = True, the returned results are smaller by a factor
      roughly equal to sample_rate, but not exactly, because the window
      correction is done differently

    With scale_by_freq = True
      The sum of the PSD is proportional to NFFT/sample_rate
      Multiplying the PSD by sample_rate/NFFT and then summing it
        gives something that is roughly equal to np.mean(signal ** 2)
      To sum up over a frequency range, could ignore NFFT and multiply
        by something like bandwidth/sample_rate, but I am not sure.
    With scale_by_freq = False
      The sum of the PSD is independent of NFFT and sample_rate
      The sum of the PSD is slightly more than np.mean(signal ** 2)
      To sum up over a frequency range, need to account for the number of
        points in that range, which depends on NFFT.
    In both cases
      The sum of the PSD is independent of the length of the signal
    The reason that the answers are not proportional to each other
    is because the window correction is done differently.

    scale_by_freq = True generally seems to be more accurate
    I imagine scale_by_freq = False might be better for quickly reading
    off a value of a peak
    """
    # Run PSD
    Pxx, freqs = matplotlib.mlab.psd(
        data,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        scale_by_freq=scale_by_freq,
        **kwargs,
    )

    # Return
    return Pxx, freqs