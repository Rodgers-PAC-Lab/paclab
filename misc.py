"""Catchall module for useful functions"""

import matplotlib.mlab
import numpy as np
import os.path
import pandas


def take_equally_spaced(arr, n):
    """Take n equally spaced elements from arr
    
    We avoid the endpoints. So, divide the array into n+1 segments, and
    take the highest point of each segment, discarding the last.
    """
    # e.g., we want 2 equally spaced, so they are at 1/3 and 2/3
    arr = np.asarray(arr)
    first_element_relative = 1.0 / (n + 1)
    relative_pos = np.linspace(
        first_element_relative, 1 - first_element_relative, n)
    absolute_pos = np.rint((len(arr) - 1) * relative_pos).astype(int)
    return arr[absolute_pos]

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


def find_ks_paths(spikesorted_dir_l, experimenter, logger_file,
                  sorted_by=np.nan):
    """
    Find the spikesorted kilosort directory for a given single sesssion.
    Modified for Rowan and Sukrith's convention of specifying who sorted a
    particular session in the metadata spreadsheet.

    Arguments
    ---
    spikesorted_dir_l : list
        A list of paths (as strings) to different users spikesorted directories on cuttlefish
    experimenter: str
        Who did the recording
    logger_file : str
        String of the current session logger file
    sorted_by : str
        Who the session was sorted by, if specified

    Returns
    ---
    kilosort_dir : str
        The path selected to use as the kilosort directory
    found_paths_l : list
        A list of the valid paths with kilosort directories for this session

    """
    # Get the correct kilosort directory,
    # along with a list of all valid paths for the kilosort directory
    # This lets you sanity check duplicate paths

    found_paths_l = []
    # Get all the valid paths
    for search_path in spikesorted_dir_l:
        test = os.path.join(search_path, logger_file)
        if os.path.exists(test):
            found_paths_l.append(test)
    # If there's no found paths, warn and exit
    if len(found_paths_l)<1:
        print(
            f'cannot find spikesorted data for '
            f'{experimenter} {mouse_name} {row} {neural_session}'
            )
        kilosort_dir = ''
        return kilosort_dir, found_paths_l

    # Add duplicates to a list, but use the path that matches experimenter
    for path in found_paths_l:
        if experimenter in path:
            kilosort_dir = path

    # If the session has an explicit 'sorted_by' given, overwrite kilosort_dir
    # to use the sorted_by person's path instead.
    try:
        np.isnan(sorted_by)
    except:
        # Fails on strings
        if type(sorted_by) == str:
        # Sorted by is a np.nan float type when it's empty so
        # make sure this is a string first
            sorted_by = sorted_by.lower()
            for search_path in found_paths_l:
                search_path_split = search_path.split('cuttlefish')[1]
                if sorted_by in search_path_split:
                    kilosort_dir = search_path

    return kilosort_dir, found_paths_l