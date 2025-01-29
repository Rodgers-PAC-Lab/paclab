"""Functions for signal processing of ABR data

"""
import scipy.signal
import numpy as np
import pandas

def identify_click_times(speaker_signal_V, threshold_V,
    highpass_freq=50, sampling_rate=16000, refrac_samples=750,
    slice_start_sample=None, slice_stop_sample=None):
    """Identify when the clicks occurred
    
    speaker_signal : array
        Voltage on the speaker, in V
    
    threshold_V : scalar
        Minimum value of a click, in V
    
    refrac_samples : scalar
        Minimum time between clicks ("refractory period") in samples
    
    slice_start_sample : int or None
        If not None, should be an integer index indicating how much data
        will be sliced before the click (example: -10). Peak times within
        -slice_start_sample of the beginning of data will be dropped.
        
    slice_stop_sample : int or None
        If not None, should be an integer index indicating how much data
        will be sliced before the click (example: +10). Peak times within
        -slice_stop_sample of the end of data will be dropped.
    
    Returns: dict with the following items
        highpass : array
            The speaker signal highpassed
        peak_time_samples : array
            Peak times in samples
        peak_properties : dict
    """
    ## Highpass audio data above 50 Hz to remove noise
    # This remove some baseline noise, but also makes the big clicks "ring"
    nyquist_freq = sampling_rate / 2.0
    ad_ahi, ad_bhi = scipy.signal.butter(
        2, (highpass_freq / nyquist_freq), btype='highpass')
    
    # Using lfilter ensures that the real peak always comes first, but
    # also introduces a delay. 
    # Using filtfilt removes the possibility of delay. 
    # In practice, they both work fine, and we are no longer using "which
    # came first" to choose peaks, but rather "which is bigger" (in find_peaks)
    speaker_signal_hp = scipy.signal.filtfilt(ad_ahi, ad_bhi, speaker_signal_V)

    # The problem with finding negative and positive peaks separately is that
    # the refractory period ("distance") is only enforced within one or the
    # other. Rather than enforce it twice, just run on the absolute value.
    #
    # height : peak must be at least this high
    # threshold : this is a criterion on diff between adjacent samples, we
    #   don't use this because it can depend on when the sample occurred
    # distance : refractory period. Smaller peaks are removed first.
    # prominence : minimum distance from height to the nearest valley within
    #   `wlen` samples. Since we use abs(), prominence will always be less than
    #   height, so we use a slightly more lenient criteria here. 
    # width : (min, max) width of the peak. We know peaks should be very
    #   narrow, no more than three samples. Due to interpolation, this is a float
    # wlen : number of samples over which to seek for a valley in calculating
    #   prominence. Since our peaks are narrow and we've highpassed, this 
    #   can be pretty short.
    # rel_height : how far the signal has to fall from peak in order to
    #   calucate "width". The thresh is peak_height - rel_height * peak_prominence
    peak_times, peak_props = scipy.signal.find_peaks(
        np.abs(speaker_signal_hp), 
        height=threshold_V, 
        distance=refrac_samples,
        prominence=threshold_V * .7,
        width=(0.5, 3.5),
        rel_height=0.5,
        wlen=20,
        )

    # Drop peak_times within left_margin or right_margin of the edge
    if slice_start_sample is not None:
        peak_times = peak_times[peak_times > -slice_start_sample]
    
    if slice_stop_sample is not None:
        peak_times = peak_times[
            peak_times < len(speaker_signal_hp) - slice_stop_sample]

    return {
        'highpassed': speaker_signal_hp,
        'peak_time_samples': peak_times,
        'peak_properties': peak_props,
        }

def categorize_clicks(click_times, speaker_signal_hp, amplitude_cuts, 
    amplitude_labels):
    """Categorize click times by amplitude
    
    click_times : array
        Obtained from identify_click_times
    
    speaker_signal_hp : array
        The highpass filtered speaker signal
        This is used to pull the amplitude of each click at its click_time
    
    amplitude_cuts : array
        The edges of the bins for each amplitude category
        These must be in the units of log(abs(speaker_signal_hp))
    
    amplitude_labels : array
        The labels to assign to each category
        len(amplitude_labels) must be len(amplitude_cuts) - 1
    
    Returns: DataFrame
        t_samples : time in samples (same as click_times)
        amplitude_V : amplitude of the click in same units as speaker_signal_hp
        amplitude_log_abs : abs(log10(amplitude_V))
        amplitude_idx : the index into amplitude_labels for this click
        polarity : True if click is positive else False
        label : the category label for the click
    """
    ## Categorize the onsets
    # Convert to series
    click_params = pandas.Series(click_times, name='t_samples').to_frame()

    # Extract the amplitude of each click
    click_params['amplitude_V'] = speaker_signal_hp[click_times]
    click_params['amplitude_log_abs'] = np.log10(
        np.abs(click_params['amplitude_V']))

    # Cut
    click_params['amplitude_idx'] = pandas.cut(
        click_params['amplitude_log_abs'], 
        bins=amplitude_cuts, labels=False)

    # Drop any that are above the last cut
    big_click_mask = (
        click_params['amplitude_log_abs'] > amplitude_cuts[-1])
    if np.any(big_click_mask):
        print('warning: dropping {} too large clicks'.format(
            big_click_mask.sum()))
        click_params = click_params[~big_click_mask]

    # After this, there should be no clicks with null amplitude_idx
    click_params['amplitude_idx'] = click_params['amplitude_idx'].astype(int)

    # Label
    click_params['label'] = amplitude_labels[
        click_params['amplitude_idx']]

    # Define polarity
    click_params['polarity'] = click_params['amplitude_V'] > 0
    
    return click_params

def slice_audio_on_clicks(speaker_signal_hp, click_params,
    slice_start_sample=-10, slice_stop_sample=10):
    """Extract a slice of audio data around each click
    
    speaker_signal_hp : array
        Audio data
    
    click_params : DataFrame, from categorize_clicks
    
    slice_start_sample, slice_stop_sample : how much data to take around click
    
    Returns: DataFrame
        index: label * polarity * t_samples
        columns: timepoint
        One row per click
    """
    # A 0.1 ms click is only a few samples long
    triggered_ad = np.array([
        speaker_signal_hp[
        trigger + slice_start_sample:trigger + slice_stop_sample]
        for trigger in click_params['t_samples']])

    # DataFrame
    triggered_ad = pandas.DataFrame(triggered_ad)
    
    # Label columns with timepoints
    triggered_ad.columns = pandas.Series(
        range(slice_start_sample, slice_stop_sample),
        name='timepoint')
    
    # Label index with click identity
    triggered_ad.index = pandas.MultiIndex.from_frame(
        click_params[['label', 'polarity', 't_samples']])
    
    # Reorder ldevels
    triggered_ad = triggered_ad.reorder_levels(
        ['label', 'polarity', 't_samples']).sort_index()
    
    return triggered_ad

def slice_neural_on_clicks(neural_data_hp, click_params, 
    slice_start_sample, slice_stop_sample, channel_names=None):
    """ Extract a slice of neural data around each click
    
    The only difference from slice_audio_on_clicks is that this works on a
    2d array instead of a 1d array.
    
    neural_data_hp : 2d array
        Audio data
    
    click_params : DataFrame, from categorize_clicks
    
    slice_start_sample, slice_stop_sample : how much data to take around click
    
    channel_names : list-like
        Must be the same length as the number of columns in neural_data_hp
        These become the names of the channels in the result
        If this is None, then the channels are simply numbered in the result
    
    Returns: DataFrame
        index: label * polarity * t_samples
        columns: channel * timepoint
        One row per click    
    """
    # Default channel_names is numbers
    if channel_names is None:
        channel_names = list(range(neural_data_hp.shape[1]))
    
    # Extract highpassed neural data around triggers
    # Shape is (n_triggers, n_timepoints, n_channels)
    triggered_neural = np.array([
        neural_data_hp[trigger + slice_start_sample:trigger + slice_stop_sample]
        for trigger in click_params['t_samples']])

    # Remove channel as a level
    triggered_neural = triggered_neural.reshape(
        [triggered_neural.shape[0], -1])        

    # DataFrame
    triggered_neural = pandas.DataFrame(triggered_neural)
    triggered_neural.index = pandas.MultiIndex.from_frame(
        click_params[['label', 'polarity', 't_samples']])
    triggered_neural = triggered_neural.reorder_levels(
        ['label', 'polarity', 't_samples']).sort_index()

    # The columns are interdigitated samples and channels
    triggered_neural.columns = pandas.MultiIndex.from_product([
        pandas.Index(
        range(slice_start_sample, slice_stop_sample), name='timepoint'),
        pandas.Index(channel_names, name='channel')
        ])
    
    # Stack channel on index, because ultimately these will be concatenated
    # over sessions, and channels won't be the same
    triggered_neural = triggered_neural.stack('channel', future_stack=True)
    
    return triggered_neural

def trim_outliers(df, abs_max_sigma=3, stdev_sigma=3):
    """Drop outlier rows from df
    
    df : DataFrame
    abs_max_sigma : Drop rows with an absmax greater than this many sigma
        from the mean
    stdev_sigma : Same for the stdev of the row
    
    Returns: DataFrame
        A DataFrame with the outlier rows dropped
    """
    res = df.copy()
    
    if abs_max_sigma is not None:
        # Find the absmax of each trial
        vals1 = res.abs().max(1)

        # Typically this is bimodal, with a normal distribution of real data
        # and a second mode contaminated by EKG
        # A simple approach is to drop trials >N sigma from the mean
        thresh1 = vals1.mean() + abs_max_sigma * vals1.std()
        mask = vals1 > thresh1
        
        # Drop
        res = res.loc[~mask]

    if stdev_sigma is not None:
        # Same with the stdev of each trial
        vals2 = res.std(1)
        thresh2 = vals2.mean() + stdev_sigma * vals2.std()
        mask = vals2 > thresh2

        # Drop
        res = res.loc[~mask]
    
    return res

def trim_outliers_from_neural(triggered_neural, abs_max_sigma=3, stdev_sigma=3):
    """Trim the outliers, separately by channel.
    
    Workflow
    ---
    Slice out each channel of triggered neural. Apply trim_outliers to it,
    which removes the rows that are outliers according to abs_max_sigma
    and stdev_sigma (see `trim_outliers`). Concatenates the results back
    together again.
    
    Returns: DataFrame
        Same columns as triggered_neural, but with the outlier rows dropped
    """
    # Iterate over channels
    trimmed_l = []
    keys_l = []
    for channel in triggered_neural.index.get_level_values('channel').unique():
        # Slice
        this_tn = triggered_neural.xs(channel, level='channel')
        
        # Trim
        trimmed = trim_outliers(
            this_tn,
            abs_max_sigma=abs_max_sigma, 
            stdev_sigma=stdev_sigma,
            )
        
        # Store
        trimmed_l.append(trimmed)
        keys_l.append(channel)
    
    # Concat
    big_trimmed = pandas.concat(trimmed_l, keys=keys_l, names=['channel'])
    
    # Reorder levels to be like triggered_neural
    big_trimmed = big_trimmed.reorder_levels(
        triggered_neural.index.names).sort_index()      

    return big_trimmed