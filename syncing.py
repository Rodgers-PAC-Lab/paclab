def get_trial_start_times(trial_start_signal, analog_fs=25000.):
    """Return the time (in seconds) of each trial start pulse"""
    # Also parse trial start signals
    # These are currently set to 100 ms, 
    # but on average are 115 ms, and can be up to 150 ms
    # 2023-07-14: Now mostly between 105 and 120 ms, with a few outliers up to 130 ms
    # 2023-10-30: a weird one at the beginning that was split into two pulses,
    #   the first of which was only 62 ms? Actually I think that's a startup pulse
    #
    # The startup pulse is a long one (6254 samples) followed very quickly
    # afterwards by a short one (1702 samples)
    # It seems like we should drop the long one and keep the short one (why?)
    trial_start_times, trial_start_pulse_duration_a = (
        MCwatch.behavior.syncing.extract_onsets_and_durations(
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
    minimum_duration_ms=5, prefilter_highpass=500):
    """Identify sound times in the actual speaker signal
    
    Returns: actual_sound_times, actual_sound_durations
        Both are in seconds
    """
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
        verbose=True, 
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

    return actual_sound_times, actual_sound_durations
