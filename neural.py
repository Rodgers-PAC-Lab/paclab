import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import my.plot

def make_plot(
    data, ax=None, n_range=None, sampling_rate=20000., 
    ch_list=None, ch_labels=None, ch_ypos=None,
    downsample=1, scaling_factor=6250./32768,
    inter_ch_spacing=234,
    max_data_size=1e7, highpass=False,
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
        the columns of `data`. This also determines the order in which they
        will be plotted (from top to bottom of the figure)
    
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
        ax.plot(t_ds, col + y_offset, 'k', lw=.75, clip_on=False)

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