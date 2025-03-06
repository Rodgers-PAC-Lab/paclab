"""Functions for syncing different types of data

Some convenience functions have been moved here from my.syncing
---
extract_onsets_and_durations - Get the start time and duration of pulses
    in a signal
drop_refrac - drop onsets too close together
extract_duration_of_onsets, extract_duration_of_onsets2 - two methods
    of getting the duration, and dropping conflicting onsets
longest_unique_fit - find the alignment between two sets of onsets

Functions for parsing our type of data
---
get_trial_start_times - Get the trial start times and durations from
    an analog trial start signal. 
get_actual_sound_times - Get the sound times from a speaker signal
"""

import scipy.signal
import scipy.stats
import pandas
import paclab
import numpy as np

def extract_onsets_and_durations(lums, delta=30, diffsize=3, refrac=5,
    verbose=False, maximum_duration=100, meth=2):
    """Identify sudden, sustained increments in the signal `lums`.
    
    meth: int
        if 2, use extract_duration_of_onsets2 
            This was the default for a long time
            This is a "greedy algorithm". 
            It prioritizes earlier onsets / longer durations
        if 1, use extract_duration_of_onsets
            It prioritizes later onsets / shorter durations
        
        The difference occurs if we have onset1 (with no matching offset1),
        followed by matching (onset2, offset2). The greedy algorithm will
        prioritize the first onset, and match offset2 to onset1, so it has
        to drop onset2. The standard algorithm will prioritize the last onset
        before the upcoming offset (i.e., prioritize shorter duration).
    
    Algorithm
    1.  Take the diff of lums over a period of `diffsize`.
        In code, this is: lums[diffsize:] - lums[:-diffsize]
        Note that this means we cannot detect an onset before `diffsize`.
        Also note that this "smears" sudden onsets, so later we will always
        take the earliest point.
    2.  Threshold this signal to identify onsets (indexes above 
        threshold) and offsets (indexes below -threshold). Add `diffsize`
        to each onset and offset to account for the shift incurred in step 1.
    3.  Drop consecutive onsets that occur within `refrac` samples of
        each other. Repeat separately for offsets. This is done with
        the function `drop_refrac`. Because this is done separately, note
        that if two increments are separated by a brief return to baseline,
        the second increment will be completely ignored (not combined with
        the first one).
    4.  Convert the onsets and offsets into onsets and durations. This is
        done with the function `extract duration of onsets2`. This discards
        any onset without a matching offset.
    5.  Drop any matched onsets/offsets that exceed maximum_duration
    
    TODO: consider applying a boxcar of XXX frames first.
    
    Returns: onsets, durations
        onsets : array of the onset of each increment, in samples.
            This will be the first sample that includes the detectable
            increment, not the sample before it.
        durations : array of the duration of each increment, in samples
            Same length as onsets. This is "Pythonic", so if samples 10-12
            are elevated but 9 and 13 are not, the onset is 10 and the duration
            is 3.
    """
    # diff the sig over a period of diffsize
    diffsig = lums[diffsize:] - lums[:-diffsize]

    # Threshold and account for the shift
    onsets = np.where(diffsig > delta)[0] + diffsize
    offsets = np.where(diffsig < -delta)[0] + diffsize
    if verbose:
        print("initial onsets")
        print(onsets)
        print("initial offsets")
        print(offsets)
    
    # drop refractory onsets, offsets
    onsets2 = drop_refrac(onsets, refrac)
    offsets2 = drop_refrac(offsets, refrac)    
    if verbose:
        print("after dropping refractory violations: onsets")
        print(onsets2)
        print("after dropping refractory violations: offsets")
        print(offsets2)
    
    # Match onsets to offsets
    if meth == 1:
        remaining_onsets, durations = extract_duration_of_onsets(onsets2, offsets2)
    elif meth == 2:
        remaining_onsets, durations = extract_duration_of_onsets2(onsets2, offsets2)
    else:
        raise ValueError("unexpected meth {}, should be 1 or 2".format(meth))
    if verbose:
        print("after combining onsets and offsets: onsets-offsets-durations")
        print(np.array([remaining_onsets, remaining_onsets + durations, durations]).T)
    
    # apply maximum duration mask
    if maximum_duration is not None:
        max_dur_mask = durations <= maximum_duration
        remaining_onsets = remaining_onsets[max_dur_mask].copy()
        durations = durations[max_dur_mask].copy()
        
        if verbose:
            print("after applying max duration mask: onsets-offsets-durations")
            print(np.array([remaining_onsets, remaining_onsets + durations, durations]).T)


    return remaining_onsets, durations
    
def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]

def extract_duration_of_onsets(onsets, offsets):
    """Extract duration of each onset.
    
    The duration is the time to the next offset. If there is another 
    intervening onset, then drop the first one.
    
    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    for idx, val in enumerate(onsets):
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Find upcoming onsets and skip if there is one before next offset
        upcoming_onsets = onsets[onsets > val]
        if len(upcoming_onsets) > 0 and upcoming_onsets[0] < next_offset:
            continue
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)    

    return np.asarray(onsets3), np.asarray(durations)

def extract_duration_of_onsets2(onsets, offsets):
    """Extract duration of each onset.
    
    Use a "greedy" algorithm. For each onset:
        * Assign it to the next offset
        * Drop any intervening onsets
        * Continue with the next onset

    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    
    if len(onsets) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # This trigger will be set after each detected duration to mask out
    # subsequent onsets greedily
    onset_trigger = np.min(onsets) - 1
    
    # Iterate over onsets
    for idx, val in enumerate(onsets):
        # Skip onsets 
        if val < onset_trigger:
            continue
        
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)
        
        # Save this trigger to skip subsequent onsets greedily
        onset_trigger = next_offset

    return np.asarray(onsets3), np.asarray(durations)

def longest_unique_fit(xdata, ydata, start_fitlen=3, ss_thresh=.0003,
    verbose=True, x_midslice_start=None, return_all_data=False,
    refit_data=False):
    """Find the longest consecutive string of fit points between x and y.

    We start by taking a slice from xdata of length `start_fitlen` 
    points. This slice is centered at `x_midslice_start` (by default,
    halfway through). We then take all possible contiguous slices of 
    the same length from `ydata`; fit each one to the slice from `xdata`;
    and calculate the best-fit sum-squared residual per data point. 
    
    (Technically seems we are fitting from Y to X.)
    
    If any slices have a per-point residual less than `ss_thresh`, then 
    increment the length of the fit and repeat. If none do, then return 
    the best fit for the previous iteration, or None if this is the first
    iteration.
    
    Usually it's best to begin with a small ss_thresh, because otherwise
    bad data points can get incorporated at the ends and progressively worsen
    the fit. If no fit can be found, try increasing ss_thresh, or specifying a
    different x_midslice_start. Note that it will break if the slice in
    xdata does not occur anywhere in ydata, so make sure that the midpoint
    of xdata is likely to be somewhere in ydata.

    xdata, ydata : unmatched data to be fit
    start_fitlen : length of the initial slice
    ss_thresh : threshold sum-squared residual per data point to count
        as an acceptable fit. These will be in the units of X.
    verbose : issue status messages
    x_midslice_start : the center of the data to take from `xdata`. 
        By default, this is the midpoint of `xdata`.
    return_all_data : boolean
        Return x_start, y_start, etc.
    refit_data : boolean, only matters if return_all_data = True
        Once the best xvy is determined, do a last refit on the maximum
        overlap of xdata and ydata.  Useful because normally execution
        stops when we run out of data (on either end) or when a bad point
        is reached. However, this will fail badly if either xdata or ydata
        contains spurious datapoints (i.e., concatenated from another 
        session).
    
    Returns: a linear polynomial fitting from Y to X.
        Or if return_all_data, also returns the start and stop indices
        into X and Y that match up. These are Pythonic (half-open).
    """
    # Choose the idx to start with in behavior
    fitlen = start_fitlen
    last_good_fitlen = 0
    if x_midslice_start is None:
        x_midslice_start = len(xdata) // 2
    keep_going = True
    best_fitpoly = None

    if verbose:
        print("begin with fitlen", fitlen)

    while keep_going:        
        # Slice out xdata
        chosen_idxs = xdata[x_midslice_start - fitlen:x_midslice_start + fitlen]
        
        # Check if we ran out of data
        if len(chosen_idxs) != fitlen * 2:
            if verbose:
                print("out of data, breaking")
            break
        if np.any(np.isnan(chosen_idxs)):
            if verbose:
                print("nan data, breaking")
            break

        # Find the best consecutive fit among onsets
        rec_l = []
        for idx in list(range(0, len(ydata) - len(chosen_idxs) + 1)):
            # The data to fit with
            test = ydata[idx:idx + len(chosen_idxs)]
            if np.any(np.isnan(test)):
                # This happens when the last data point in ydata is nan
                continue
            
            # fit
            fitpoly = np.polyfit(test, chosen_idxs, deg=1)
            fit_to_input = np.polyval(fitpoly, test)
            resids = chosen_idxs - fit_to_input
            ss = np.sum(resids ** 2)
            rec_l.append({'idx': idx, 'ss': ss, 'fitpoly': fitpoly})
        
        # Test if there were no fits to analyze
        if len(rec_l) == 0:
            keep_going = False
            if verbose:
                print("no fits to threshold, breaking")
            break
        
        # Look at results
        rdf = pandas.DataFrame.from_records(rec_l).set_index('idx').dropna()

        # Keep only those under thresh
        rdf = rdf[rdf['ss'] < ss_thresh * len(chosen_idxs)]    

        # If no fits, then quit
        if len(rdf) == 0:
            keep_going = False
            if verbose:
                print("no fits under threshold, breaking")
            break
        
        # Take the best fit
        best_index = rdf['ss'].idxmin()
        best_ss = rdf['ss'].min()
        best_fitpoly = rdf['fitpoly'].loc[best_index]
        if verbose:
            fmt = "fitlen=%d. best fit: x=%d, y=%d, xvy=%d, " \
                "ss=%0.3g, poly=%0.4f %0.4f"
            print(fmt % (fitlen, x_midslice_start - fitlen, best_index, 
                x_midslice_start - fitlen - best_index, 
                best_ss // len(chosen_idxs), best_fitpoly[0], best_fitpoly[1]))

        # Increase the size
        last_good_fitlen = fitlen
        fitlen = fitlen + 1    
    
    # Always return None if no fit found
    if best_fitpoly is None:
        return None
    
    if return_all_data:
        # Store results in dict
        fitdata = {
            'x_start': x_midslice_start - last_good_fitlen,
            'x_stop': x_midslice_start + last_good_fitlen,
            'y_start': best_index,
            'y_stop': best_index + last_good_fitlen * 2,
            'best_fitpoly': best_fitpoly,
            'xdata': xdata,
            'ydata': ydata,
        }            
        
        # Optionally refit to max overlap
        if refit_data:
            fitdata = refit_to_maximum_overlap(xdata, ydata, fitdata)
        
        return fitdata
    else:
        return best_fitpoly

def get_trial_start_times(
    trial_start_signal, 
    delta=5000,
    analog_fs=25000.,
    maximum_duration_samples=10000,
    verbose=False,
    ):
    """Return the time (in seconds) of each trial start pulse
    
    The trial start is defined as the onset of a trial start pulse on the
    trial start signal line in the analog data. This same pulse triggers
    an LED to flash in the arena.
    
    This function is just a wrapper around extract_onsets_and_durations with 
    defaults that make sense for trial start pulses.

   
    trial_start_signal : array
        From the analog data. Units should be bitlevels. 
    
    delta : numeric
        A threshold in bitlevels. For our current settings with the eCube,
        a 3.3 V signal tends to end up at 11000 bitlevels, which I assume is
        2 ** 15 * 3.3 / 10. So a `delta` of half that is appropriate.
    
    maximum_duration : numeric
        A maximum duration in samples of a pulse. 
        Pulses longer than this will be silently dropped.
        The pulse duration has changed over time, from 100 to 300 ms or so.
        There is no harm in making this plenty long since erroneously long
        pulses are rare (?), although this used to happen with Autopilot
        during the startup sequence.
        With Octopilot, these pulses are pretty consistently 100 ms long,
        with stdev 0.3 ms or so.
    
    verbose : bool
        Passed to extract_onsets_and_durations
    
    Returns: pulse_time_s, pulse_durations_s
        Both are arrays of the same length. Both are in units of seconds.
        An old version of this function used to return only pulse_times_s.
        pulse_times_s : start times of trials (onset of pulse)
        pulse_durations_s : duration of each trial start pulse
    """
    # Here are some old notes that were relevant only to Autopilot
    # These are currently set to 100 ms, 
    # but on average are 115 ms, and can be up to 150 ms
    # or 250 ms in the case of 2023-12-07_15-37-49
    # 2023-07-14: Now mostly between 105 and 120 ms, with a few outliers up to 130 ms
    # 2023-10-30: a weird one at the beginning that was split into two pulses,
    #   the first of which was only 62 ms? Actually I think that's a startup pulse
    #
    # The startup pulse is a long one (6254 samples) followed very quickly
    # afterwards by a short one (1702 samples)
    # It seems like we should drop the long one and keep the short one (why?)
    
    
    # Get the start times and durations
    pulse_times_samples, pulse_durations_samples = (
        extract_onsets_and_durations(
        trial_start_signal, 
        delta=delta, 
        verbose=verbose, 
        maximum_duration=maximum_duration_samples,
        ))

    # More old notes relevant only to autopilot
    #~ # Edge case: I think often the first one is too short
    #~ if trial_start_pulse_duration_a[0] < 2000:
        #~ print("warning: dropping first trial start pulse")
        #~ trial_start_times = trial_start_times[1:]
        #~ trial_start_pulse_duration_a = trial_start_pulse_duration_a[1:]
    
    #~ assert (trial_start_pulse_duration_a > 2000).all()
    #~ assert (trial_start_pulse_duration_a < 3800).all()
    
    # Convert to seconds
    pulse_times_s = pulse_times_samples / analog_fs
    pulse_durations_s = pulse_durations_samples / analog_fs
    
    return pulse_times_s, pulse_durations_s

def get_video_start_time(video_save_signal, multiple_action='error'):
    """Return the time (in samples) of the video save pulse
    
    The video save signal in Room C is frequently incorrect. It's supposed
    to be high throughout the recording. Instead, it is frequently pulsed
    high for just 5 ms or so, sometimes multiple times, or never goes
    high at all. I don't know if the 5 ms pulse actually maps onto the
    start of the video or not. This might be because the saving command
    is only guaranteed for the control camera, or something like that.
    
    trig_signal : array-like
        The trigger signal
    
    multiple_action : string or None
        Controls what happens if multiple triggers detected
        'error': raise ValueError
        'warn' or 'warning': print warning
        anything else : do nothing

    Returns: start_times, durations
        start_times: start time in  samples
        durations: duration in samples
            
        If multiple triggers are detected, they are all returned in an array
        If only one, then only that one is returned
    """
    # Find threshold crossings
    # 10.0V = 32768 (I think?), so 3.3V = 10813
    # Take the first sample that exceeds roughly half that
    trig_time_a, trig_duration_a = (
        extract_onsets_and_durations(
        video_save_signal, delta=5000, verbose=False, maximum_duration=np.inf))
    
    if len(trig_time_a) != 1:
        if multiple_action == 'error':
            raise ValueError("expected 1 trig, got {}".format(len(trig_time_a)))
        elif multiple_action in ['warn', 'warning']:
            print("warning: expected 1 trig, got {}".format(len(trig_time_a)))

    if len(trig_time_a) == 1:
        return trig_time_a[0], trig_duration_a[0]
    else:
        return trig_time_a, trig_duration_a

def get_recording_start_time(trig_signal, multiple_action='error'):
    """Return the time (in samples) of the recording start pulse
    
    trig_signal : array-like
        The trigger signal
    
    multiple_action : string or None
        Controls what happens if multiple triggers detected
        'error': raise ValueError
        'warn' or 'warning': print warning
        anything else : do nothing
    
    An error occurs if this trigger is too short or too long
    
    If multiple triggers are detected, they are all returned in an array
    If only one, then only that one is returned
    """
    # Find threshold crossings
    # 10.0V = 32768 (I think?), so 3.3V = 10813
    # Take the first sample that exceeds roughly half that
    # We expect trig signal to last 100 ms (I think?), which is 2500 samples
    # There is a pulse about 6000 samples long at the very beginning, which I
    # think is when the nosepoke is initialized
    trig_time_a, trig_duration_a = (
        extract_onsets_and_durations(
        trig_signal, delta=5000, verbose=False, maximum_duration=5000))
    
    if len(trig_time_a) != 1:
        if multiple_action == 'error':
            raise ValueError("expected 1 trig, got {}".format(len(trig_time_a)))
        elif multiple_action in ['warn', 'warning']:
            print("warning: expected 1 trig, got {}".format(len(trig_time_a)))

    assert (trig_duration_a > 2495).all()
    assert (trig_duration_a < 2540).all()

    if len(trig_time_a) == 1:
        return trig_time_a[0]
    else:
        return trig_time_a

def get_actual_sound_times(speaker_signal, threshhold=50, analog_fs=25000.,
    minimum_duration_ms=5, prefilter_highpass=500, return_as_dict=True,
    verbose=True):
    """Identify sound times in the actual speaker signal
    
    speaker_signal : raw data from speaker
    threshold : float
        Always overwrites the od threshold manually
    analog_fs : sampling rate of speaker_signal
    minimum_duraion_ms: shorter than this is discarded
    prefilter_highpass : float
    return_as_dict : bool
        if True, return data as dict
        False is obsolete
    
    Returns: dict
        'onset_times': onset times in seconds
        'durations': in seconds
        'od': the OnsetDetector object

    """
    # Annoying dependency
    import ns5_process.AudioTools
    
    # Identify the onsets by squaring and smoothing
    # Typical sound is about 320 samples long, or 13 ms
    bhi, ahi = scipy.signal.butter(
        2, prefilter_highpass / (analog_fs / 2), 'high')

    # hp filter
    hp = scipy.signal.filtfilt(bhi, ahi, speaker_signal)

    # find onsets and offsets
    # note that smoothing length is hardcoded (??!) as 3 ms in here
    od = ns5_process.AudioTools.OnsetDetector(
        hp, 
        F_SAMP=analog_fs, 
        verbose=verbose, 
        plot_debugging_figures=False,
        minimum_threshhold=None,
        minimum_duration_ms=minimum_duration_ms,
        )

    # This one works better
    od.thresh_setter = ns5_process.AudioTools.ThreshholdAutosetterMinimalHistogram

    # Manual
    od.threshhold = threshhold

    # Run onset detector
    od.execute()

    # Calculate sound times
    actual_sound_times = od.detected_onsets / analog_fs
    actual_sound_durations = (od.detected_offsets - od.detected_onsets) / analog_fs
    
    if return_as_dict:
        return {
            'onset_times': actual_sound_times,
            'durations': actual_sound_durations,
            'od': od,
            }
    else:
        return actual_sound_times, actual_sound_durations
    
def sync_behavior_to_analog(
    session_trial_data,
    session_sound_data,
    h5_filename,
    analog_packed_filename,
    drop_first_n_analog_triggers=None,
    analog_fs=25000.,
    ):
    """Sync session
    
    This is a legacy function that only works with Autopilot style data.
    
    session_trial_data : from paclab.parse.load_data
    session_sound_data : from paclab.parse.load_data
    h5_filename : full path to hdf5 file
    analog_packed_filename : full path to analog file
    drop_first_n_analog_triggers : how many analog triggers to drop

    Fits the flash time on rpi01 according to the rpi01 clock to the 
    detected flash time on the analog input. This should be a pretty good
    way of converting rpi01 clock time to analog time, up to the limit
    set by how quickly the DIO can be triggered, which seems to be ~1 ms
    jitter (and unknown offset).

    Returns: dict
        'b2a_slope': behavior2analog_fit_rpi01.slope,
        'b2a_intercept': behavior2analog_fit_rpi01.intercept,
        'b2a_rval': 1 - behavior2analog_fit_rpi01.rvalue,
        'n_trials_behavior': len(session_trial_data),
        'n_trials_analog': len(analog_trial_start_times_s),
        'neural_start_time_analog_samples': trig_time,
        'video_start_time_analog_samples': video_start_time_analog_samples,
        'video_duration': video_duration,
        'resids': resids,    
    """
    # Define this as the session start time
    session_start_time = session_trial_data['trial_start'].iloc[0]

    # Normalize times in session_trial_data to the session_start
    # Calculate the relative time between each trial start and the session start
    # These are all in parent pi time
    session_trial_data['trial_start_time_in_session'] = (
        session_trial_data['trial_start'] - session_start_time
        ).apply(lambda dt: dt.total_seconds())


    ## Load flash times from HDF5 file
    # Load flash times
    # This is dt_flash_received, so in child pi time
    flash_df = paclab.parse.load_flash_df(h5_filename)

    # Add the trial start, according to the parent
    # This is missing on the last trial, which is always dropped
    flash_df = flash_df.join(session_trial_data['trial_start'])

    # Calculate the relative time between each rpi flash and the session start
    # This ASSUMES that child pi time and parent pi time are exactly matched,
    # which is only true up to the resolution of chrony
    flash_df_wrt_session_start = (flash_df - session_start_time).apply(
        lambda ser: ser.dt.total_seconds())


    ## Load SoundsPlayed from HDF5 file name
    # These are in "speaker_time_in_session", which is in child pi time, relative
    # to the value `session_start_time`
    # So subject to the same caveat as rpi01_flash_time_in_session
    # But should be directly comparable to rpi01_flash_time_in_session
    sounds_played_df = paclab.parse.load_sounds_played(
        h5_filename, session_start_time)


    ## Load analog data
    # Memmap
    analog_mm = paclab.neural.load_analog_data(analog_packed_filename)

    # Extract needed data
    trig_signal = analog_mm[:, 0]
    trial_start_signal = analog_mm[:, 1]
    video_save_signal = analog_mm[:, 4]

    # Detect when video save began
    # This is very often wrong, but keep it anyway
    video_start_time_analog_samples, video_duration = (
        paclab.neural.get_video_start_time(
        video_save_signal, multiple_action='ignore')
        )
    
    # Detect trial starts in trig_signal and trial_start_signal
    # This is when the neural recording started
    trig_time = paclab.neural.get_recording_start_time(
        trig_signal, multiple_action='warn')

    # This is when each trial started, according to the time that the
    # nosepoke LED flashed on the single recorded port 
    analog_trial_start_times_s = get_trial_start_times(
        trial_start_signal)
    
    # Optionally drop the first analog triggers
    if not pandas.isnull(drop_first_n_analog_triggers):
        print("dropping first {} analog triggers".format(
            int(drop_first_n_analog_triggers)))
        analog_trial_start_times_s = analog_trial_start_times_s[
            int(drop_first_n_analog_triggers):]
    
    # Warn if the analog data seems too short
    if len(analog_mm) < 60 * analog_fs:
        print("warning: analog data is only {:.1f} s long".format(
            len(analog_mm) / analog_fs))

    if len(analog_trial_start_times_s) <= 2:
        print("warning: only {} analog starts detected".format(
            len(analog_trial_start_times_s)))

    if len(session_trial_data) != len(analog_trial_start_times_s) - 1:
        print("warning: " +
            "{} completed trials and {} analog starts detected".format(
            len(session_trial_data),
            len(analog_trial_start_times_s) - 1
            ))

    # Add analog triggers to session_trial_data
    # The last trial start NEVER included, because last trial always incomplete
    assert len(session_trial_data) == len(analog_trial_start_times_s) - 1
    session_trial_data['analog_trial_start'] = analog_trial_start_times_s[:-1]


    ## Align the trial_start time with the analog trial_start_time
    # Fit rpi01_flash_time_in_session (the time that rpi01 thinks it flashed)
    # to analog_trial_start (the time that the flash was recorded by eCube)
    #
    # This is the fit we actually use to predict when sound was played
    # To make predictions for other speakers, we'd have to assume that they have
    # a fixed latency wrt rpi01, which is only sort of true.
    rpi01_flash_time = session_trial_data.join(
        flash_df_wrt_session_start['rpi01'])['rpi01'].values
    analog_trial_start_time = session_trial_data['analog_trial_start'].values

    # Do the fit
    behavior2analog_fit_rpi01 = scipy.stats.linregress(
        rpi01_flash_time, analog_trial_start_time,
        )
    
    # Calculate the resids
    analog_pred_from_behavior = np.polyval(
        [behavior2analog_fit_rpi01.slope, behavior2analog_fit_rpi01.intercept], 
        rpi01_flash_time)
    resids = analog_trial_start_time - analog_pred_from_behavior
    
    # These should ideally be within 1 ms because the flash should go off
    # just at the time that the trial start signal is received
    # In fact, the std is 0.7 ms with a long tail out to 3 ms
    # Probably due to DIO jitter

    
    ## Return
    return {
        'b2a_slope': behavior2analog_fit_rpi01.slope,
        'b2a_intercept': behavior2analog_fit_rpi01.intercept,
        'b2a_rval': 1 - behavior2analog_fit_rpi01.rvalue,
        'n_trials_behavior': len(session_trial_data),
        'n_trials_analog': len(analog_trial_start_times_s),
        'neural_start_time_analog_samples': trig_time,
        'video_start_time_analog_samples': video_start_time_analog_samples,
        'video_duration': video_duration,
        'resids': resids,
        }

def fit_analog_flash_to_behavior_flash(
    trials, 
    flash_time_behavior_s,
    flash_time_analog_s,
    std_resids_thresh=0.001,
    max_resids_thresh=0.01,
    ):
    """Fit behavior to analog
    
    trials : DataFrame, indexed by trial_number
    
    flash_time_behavior_s : Series, indexed by trial number
        The flash time reported by the rpi that we recorded in the analog data.
        These times should be in the behavioral timebase: seconds since
        the start of the first trial, on that pi.
        If this doesn't align with the `trials` exactly, a warning is issued.
    
    flash_time_analog_s : array
        The flash time recorded in the analog data, in seconds since
        the beginning of the analog recording.
        This must have either the same length as `trials` or one greater than
        the length of `triasl`.
    
    std_resids_thresh, max_resids_thresh : numeric
        If the fit residuals exceed these metrics, a warning is printed
    
    Workflow:
    * Store flash_time_analog_s as trials['flash_time_analog_s']
        Since we have no trial number information in this array, we assume
        that the first recorded analog flash is the first trial. The last
        recorded analog flash must be the last behavioral trial or one after
        the last behavioral trial (in case that behavioral trial was incomplete
        and therefore missing from `trials`.)
    * Store flash_time_behavior_s as trials['flash_time_behavior_s'].
        If it doesn't align perfectly, a warning is issued and those trials
        are dropped.
    * Fit between trials['flash_time_analog_s'] and 
        trials['flash_time_behavior_s']. This tells us the relationship
        between analog time and behavior time on the recorded rpi. For now,
        we have to assume that all other rpis have the same behavioral timebase,
        which is only true up to the limits of chrony. Eventually, we need
        to replace this with some kind of hardware measurement on each pi.
    
    Empirically, the fit had a stdev of 0.7 ms with autopilot and 0.3 ms
    with octopilot. This is presumably set by DIO jitter.
    TODO: try RPi.GPIO here instead
    TODO: loopback on the trial_start pin, to upper bound how long it takes    
    
    Returns: dict, with keys
        'trials': new version of trials with the columns above added
            and potentially rows dropped
        'b2a_slope', 'b2a_intercept': the fit from behavior to analog
    """
    ## Make a copy
    trials = trials.copy()
    
    
    ## First, store the analog trial start times in `trials`
    # TODO: replace this with longest_unique_fit on trials['start_time'], 
    # in case the analog recording is incomplete
    # For now, we assume that analog data includes all trials.
    
    if len(trials) == len(flash_time_analog_s):
        # One behavior trial per analog start
        trials['flash_time_analog_s'] = flash_time_analog_s

    elif len(trials) == len(flash_time_analog_s) - 1:
        # One fewer behavior trial than analog trial
        # This is not uncommon, because the last behavior trial is often 
        # incomplete, meaning it won't appear in trial_data
        trials['flash_time_analog_s'] = flash_time_analog_s[:-1]

    else:
        print('error: {} behavior trials but {} analog trials'.format(
            len(trials), len(flash_time_analog_s)))
    
    
    ## Next, store the reported flash times in `trials`
    trials = trials.join(
        flash_time_behavior_s.rename('flash_time_behavior_s'))
    
    # Warn on any null
    # TODO: does this ever actually happen?
    null_mask = trials['flash_time_behavior_s'].isnull()
    if null_mask.any():
        print(
            'warning: dropping trials {} without behavioral flash time'.format(
            null_mask.index[null_mask.values]))
        trials = trials.dropna(subset=['flash_time_behavior_s'])

    
    ## Do the fit
    # TODO: replace with longest_unique_fit
    behavior2analog_fit_rpi01 = scipy.stats.linregress(
        trials['flash_time_behavior_s'].values, 
        trials['flash_time_analog_s'].values,
        )
    
    
    ## Calculate the residuals
    analog_pred_from_behavior = np.polyval(
        [behavior2analog_fit_rpi01.slope, behavior2analog_fit_rpi01.intercept], 
        flash_time_behavior_s.values)
    resids = flash_time_analog_s - analog_pred_from_behavior

    # Warn if these metrics are broken
    std_resids = np.std(resids)
    if std_resids > std_resids_thresh:
        print(
            f'warning: stdev(behavior2analog resids) is {std_resids}, '
            f'exceeding thresh of {std_resids_thresh}'
            )

    max_resids = np.max(np.abs(resids))
    if max_resids > max_resids_thresh:
        print(
            f'warning: absmax(behavior2analog resids) is {max_resids}, '
            f'exceeding thresh of {max_resids_thresh}'
            )

    
    ## Return
    return {
        'trials': trials,
        'b2a_slope': behavior2analog_fit_rpi01.slope,
        'b2a_intercept': behavior2analog_fit_rpi01.intercept,
        }

def compose_fit(p1, p2):
    """Return a linear poly p3 s.t. p3(x) = p1(p2(x))
    
    p1, p2 : array-like of length 2
        Slope is always first, intercept always last
    
    Returns: array of length 2
    """
    return np.array([
        p1[0] * p2[0], # m1 * m2
        p1[0] * p2[1] + p1[1], # m1 * b2 + b1
        ])

def invert_fit(p):
    """Return a linear poly q such that q(p(x)) = x
    
    p : array-like of length 2
        Slope is always first, intecept always last
    
    Returns: array of length 2
    """
    return np.array([
        1 / p[0], # slope
        -p[1] / p[0], # intercept
        ])
