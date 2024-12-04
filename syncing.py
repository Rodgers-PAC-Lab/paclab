import scipy.signal
import scipy.stats
import my
import pandas
import paclab
import numpy as np

def get_trial_start_times(trial_start_signal, analog_fs=25000.):
    """Return the time (in seconds) of each trial start pulse"""
    # Also parse trial start signals
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
    trial_start_times, trial_start_pulse_duration_a = (
        my.syncing.extract_onsets_and_durations(
        trial_start_signal, delta=5000, verbose=False, maximum_duration=5000))
    
    #~ # Edge case: I think often the first one is too short
    #~ if trial_start_pulse_duration_a[0] < 2000:
        #~ print("warning: dropping first trial start pulse")
        #~ trial_start_times = trial_start_times[1:]
        #~ trial_start_pulse_duration_a = trial_start_pulse_duration_a[1:]
    
    #~ assert (trial_start_pulse_duration_a > 2000).all()
    #~ assert (trial_start_pulse_duration_a < 3800).all()
    analog_trial_start_times_s = trial_start_times / analog_fs
    
    return analog_trial_start_times_s

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