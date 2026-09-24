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

def find_kilosort_directories(spikesorted_dir_l, combined_sheet, specify_sorted_by=False):
    """
    Find the spikesorted kilosort directory for all sesssions. Copied from
    Chris' code in 20260721_sync_octagon and modified for Rowan and Sukrith's
    convention of specifying who sorted a particular session in the metadata spreadsheet.

    Arguments
    ---
    spikesorted_dir_l : list
        A list of paths (as strings) to different users spikesorted directories on cuttlefish
    combined_sheet: pandas.DataFrame
        Dataframe combined from all metadata google sheets.
        Indexed like ['experimenter', 'mouse_name', 'row']
        Columns must include logger_file, kilosort_folder and manually_sorted.
        If specify_sorted_by=True, also must include column 'sorted_by'
    specify_sorted_by: bool
        T/F, whether there's a 'sorted_by' column in the metadata

    """

    session_paths_l = []
    keys_l = []
    all_l = []
    for (experimenter, mouse_name, row) in combined_sheet.index:
        row_metadata = combined_sheet.loc[experimenter,mouse_name,row]
        neural_session = row_metadata['logger_file']

        ## Find the spikesorted_dir
        # Search each location
        found = None

        # Start with the one matching the experimenter
        default_search_path = None
        for search_path in spikesorted_dir_l:
            # Path has expanded user, so split path on 'cuttlefish' and
            # only search for the experimenter in the path after cuttlefish
            if experimenter in search_path.split('cuttlefish')[1]:
                # Keep track of the default search path
                default_search_path = search_path

                # Test
                test = os.path.join(search_path, neural_session)
                if os.path.exists(test):
                    found = test

        # Now test all of them
        for search_path in spikesorted_dir_l:
            # Skip the default since it was already tested
            if search_path == default_search_path:
                continue

            # Otherwise test
            test = os.path.join(search_path, neural_session)
            if os.path.exists(test):
                if found is None:
                    found = test
                else:
                    # Warn but do not overwrite
                    # These are likely when one person tested another's session
                    print(
                        f'duplicate kilosort path found:\n'
                        f'  v1: {found}\n  v2: {test}'
                    )

                    if specify_sorted_by:
                        # Use the path that matches the 'sorted_by' field
                        sorted_by = row_metadata['sorted_by']
                        # Sorted by is a np.nan float type when it's empty so
                        # make sure this is a string first
                        if type(sorted_by) == str:
                            sorted_by = sorted_by.lower()
                            if sorted_by in test.split('cuttlefish')[1]:
                                found = test

        # Check
        if found is None:
            print(
                f'cannot find spikesorted data for '
                f'{experimenter} {mouse_name} {row} {neural_session}'
            )
            found = ''

        # Rename for downstream code
        # kilosort_dir = found
        session_paths_l.append([experimenter,mouse_name,row,found])

    # Found paths to a dataframe
    session_paths_df = pandas.DataFrame(session_paths_l)
    session_paths_df = session_paths_df.rename(
        columns={0: 'experimenter', 1: 'mouse_name', 2: 'row', 3: 'kilosort_dir'})
    session_paths_df = session_paths_df.set_index(['experimenter', 'mouse_name', 'row'])
    return session_paths_df


def find_single_ksdir(spikesorted_dir_l, experimenter, logger_file, sorted_by=None):
    """
    Find the spikesorted kilosort directory for a given single sesssion. Copied from
    Chris' code in 20260721_sync_octagon and modified for Rowan and Sukrith's
    convention of specifying who sorted a particular session in the metadata spreadsheet.

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

    """

    # for (experimenter, mouse_name, row) in combined_sheet.index:
    #     row_metadata = combined_sheet.loc[experimenter, mouse_name, row]
    #     neural_session = row_metadata['logger_file']

    neural_session = logger_file
    ## Find the spikesorted_dir
    # Search each location
    found = None

    # Start with the one matching the experimenter
    default_search_path = None
    for search_path in spikesorted_dir_l:
        # Path has expanded user, so split path on 'cuttlefish' and
        # only search for the experimenter in the path after cuttlefish
        if experimenter in search_path.split('cuttlefish')[1]:
            # Keep track of the default search path
            default_search_path = search_path

            # Test
            test = os.path.join(search_path, neural_session)
            if os.path.exists(test):
                found = test

    # Now test all of them
    for search_path in spikesorted_dir_l:
        # Skip the default since it was already tested
        if search_path == default_search_path:
            continue

        # Otherwise test
        test = os.path.join(search_path, neural_session)
        if os.path.exists(test):
            if found is None:
                found = test
            else:
                # Warn but do not overwrite
                # These are likely when one person tested another's session
                print(
                    f'duplicate kilosort path found:\n'
                    f'  v1: {found}\n  v2: {test}'
                )

                if sorted_by != None:
                    # Use the path that matches the 'sorted_by' field
                    # Sorted by is a np.nan float type when it's empty so
                    # make sure this is a string first
                    if type(sorted_by) == str:
                        sorted_by = sorted_by.lower()
                        if sorted_by in test.split('cuttlefish')[1]:
                            found = test
                    else:
                        print("You gave a sorted_by argument that wasn't a string")

    # Check
    if found is None:
        print(
            f'cannot find spikesorted data for '
            f'{experimenter} {mouse_name} {row} {neural_session}'
        )
        found = ''

    # Rename for downstream code
    session_path = found
    return session_path