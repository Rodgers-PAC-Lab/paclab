"""Functions for loading and synchronizing behavioral, neural, video data"""

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import my
import my.plot

def load_analog_data(analog_packed_filename):
    """Check filesize on analog_packed_filename and return memmap
    
    At present:
    trig_signal = analog_mm[:, 0]
    trial_start_signal = analog_mm[:, 1]
    speaker_signal = analog_mm[:, 2]
    video_sync_signal = analog_mm[:, 3]
    video_save_signal = analog_mm[:, 4]    
    """
    # Check size
    analog_packed_file_size_bytes = os.path.getsize(analog_packed_filename)
    analog_packed_file_size_samples = analog_packed_file_size_bytes // 32 // 2
    assert analog_packed_file_size_samples * 32 * 2 == analog_packed_file_size_bytes

    # Memmap the analog data
    analog_mm = np.memmap(
        analog_packed_filename, 
        np.int16, 
        'r', 
        offset=0,
        shape=(analog_packed_file_size_samples, 32)
    )    
    
    return analog_mm

def load_neural_data(neural_packed_filename):
    packed_file_size_bytes = os.path.getsize(neural_packed_filename)
    packed_file_size_samples = (packed_file_size_bytes - 8) // 64 // 2
    assert packed_file_size_samples * 64 * 2 + 8 == packed_file_size_bytes
    neural_mm = np.memmap(
        neural_packed_filename, 
        np.int16, 
        'r', 
        offset=8,
        shape=(packed_file_size_samples, 64)
    )   
    
    return neural_mm

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
        my.syncing.extract_onsets_and_durations(
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
        my.syncing.extract_onsets_and_durations(
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

def make_plot(
    data, ax=None, n_range=None, sampling_rate=20000., 
    ch_list=None, ch_labels=None, ch_ypos=None,
    downsample=1, scaling_factor=6250./32768,
    inter_ch_spacing=234,
    max_data_size=1e7, highpass=False,
    spike_samples=None,
    spike_channels=None,
    spike_colors=None,
    plot_kwargs={},
    ):
    """Plot a vertical stack of channels in the same ax.
    
    The given time range and channel range is extracted. Optionally it
    can be filtered at this point. 
    
    data : array, shape (N_timepoints, N_channels)
    
    ax : axis to plot into. 
        If None, a new figure and axis will be created
    
    n_range : tuple (n_start, n_stop)
        The index of the samples to include in the plot
        Default is (0, len(data))
    
    ch_list : a list of the channels to include, expressed as indices into
        the columns of `data`. 
    
    ch_labels : list, same length as ch_list
        Each trace will be labeled
    
    ch_ypos : list, same length as ch_list
        The order they will be plotted in, ie the one with lowest ch_ypos
        will be plotted at the top
    
    exclude_ch_list : remove these channels from `ch_list`
    
    downsample : downsample by this factor
    
    scaling_factor : multiple by this
    
    inter_ch_spacing : channel centers are offset by this amount, in the
        same units as the data (after multiplication by scaling_factor)
    
    legend_t_offset, legend_y_offset, legend_t_width : where to plot legend
    
    max_data_size : sanity check, raise error rather than try to plot
        more than this amount of data
    
    highpass : None or float
        If float, highpass above this value
        If None, do not filter
    
    plot_kwargs : a dict to pass to `plot`, containing e.g. linewidth
    
    spike_samples, spike_channels : array-like
        If either is None, nothing happens
        These should be the same length
        These are the times of spikes (in samples) and the channels of 
        each spike (expressed as indices into ch_list).
        They will be overplotted in red. 
    """
    # data range in seconds
    t = np.arange(n_range[0], n_range[1]) / sampling_rate
    t_ds = t[::downsample]

    # If too much data is requested, then break
    got_size = len(t_ds) * len(ch_list)
    print("getting %g datapoints..." % got_size)
    if len(t_ds) * len(ch_list) > max_data_size:
        raise ValueError(
            ("you requested %g datapoints " % got_size) +
            ("which is more than max_data_size = %g" % max_data_size) +
            ("\nRequest less data or increase max_data_size")
        )
    
    # Grab the data
    # This takes a window of data, downsamples it, and orders like ch_list
    got_data = data[n_range[0]:n_range[1]:downsample, ch_list]
    got_data = got_data * scaling_factor
    
    # Optionally filter
    if highpass is not None:
        # Normally it would be highpass / sampling_rate
        # But we have alreayd downsampled by downsample, so the sampling rate
        # of got_data is (sampling_rate / downsample)
        # So use highpass / (sampling_rate / downsample)
        critical_freq = highpass / (sampling_rate / downsample)
        if critical_freq > 1:
            raise ValueError(
                "cannot apply highpass of {} with downsample of {}".format(
                highpass, downsample) + 
                "\nRemove highpass, lower highpass, or lower downsample")
        
        # Make and apply filter
        buttb, butta = scipy.signal.butter(3, critical_freq,'high')
        got_data = scipy.signal.filtfilt(buttb, butta, got_data, axis=0)
    
    # Set up ax
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 8))
        f.subplots_adjust(left=.05, right=.98, bottom=.05, top=.96)
    
    # Plot each channel
    for ncol, col in enumerate(got_data.T):
        ypos = ch_ypos[ncol]
        y_offset = -inter_ch_spacing * ypos
        ax.plot(t_ds, col + y_offset, 'k', clip_on=False, **plot_kwargs)

    
    ## Overplot the spikes
    if spike_samples is not None and spike_channels is not None:
        # Convert to int
        spike_channels = spike_channels.astype(int)
        spike_samples = spike_samples.astype(int)
        assert len(spike_channels) == len(spike_samples)
        
        # Iterate over every spike
        zobj = zip(spike_samples, spike_channels, spike_colors)
        for spike_sample, spike_channel, spike_color in zobj:
            # Get the time of this spike relative to got_data
            n_spike = (spike_sample - n_range[0]) // downsample
            n_spike_start = n_spike - (20 // downsample)
            n_spike_stop = n_spike + (20 // downsample)
            
            # Convert to real time
            t_spike_range = t_ds[n_spike_start:n_spike_stop]
            
            # Grab a slice of data
            y_spike = got_data[n_spike_start:n_spike_stop, spike_channel]
            
            # Where to plot it
            y_offset = -inter_ch_spacing * ch_ypos[spike_channel]
            
            # Plot it
            ax.plot(t_spike_range, y_spike + y_offset, color=spike_color, 
                lw=1)


    ## Pretty
    # title
    ax.set_title('distance between channels = {} uV'.format(inter_ch_spacing))

    # Y-axis
    ax.set_ylim((-inter_ch_spacing * len(ch_list), 2 * inter_ch_spacing))
    
    # Y-ticks: one per channel, labeled with channel number
    ax.set_yticks(-inter_ch_spacing * np.array(ch_ypos))
    ax.set_yticklabels(ch_labels, size='xx-small')
    
    # Y-label
    ax.set_ylabel('channels in order of ch_list')
    
    # X-ticks
    ax.set_xlabel('time (s)')
    ax.set_xlim((t[0], t[-1]))
    
    # Pretty
    my.plot.despine(ax)
    
    return {
        'ax': ax,
        'data': got_data,
        }
