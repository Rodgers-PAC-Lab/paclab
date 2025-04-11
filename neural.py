"""Functions for loading and synchronizing behavioral, neural, video data"""

import os
import datetime
import xml.etree.ElementTree
import glob
import numpy as np
import pandas
import scipy
import matplotlib.pyplot as plt
import my
import my.plot

def find_logger(neural_root, session_name):
    """Determine which logger was used for a certain session name.
    
    Arguments:
    ---
    neural_root : str
        A full path to 20241125_HSW, or whichever directory contains the
        logger files.
    
    session_name : str
        A session name, without '.bin'
    
    Flow
    ---
    * Finds all subdirectories of `neural_root` that begin with "logger_"
    * Checks each one for the file `session_name` + ".bin"
    * Raises IOError if it's found in multiple logger subdirectories
    * Raises IOError if session isn't found anywhere
    
    Returns: str or None
        The logger name if it was found, or None if the session wasn't found
        This will be like '628DAB' not 'logger_628DAB'
    """
    # Find all logger directories
    logger_dirs = glob.glob(os.path.join(neural_root, 'logger_*'))
    
    # Find out which one contains the session
    logger = None
    for test_logger_dir in logger_dirs:
        # Get logger name from the logger directory name
        test_logger_name = os.path.split(test_logger_dir)[1].replace(
            'logger_', '')
        
        # See if this session exists
        if os.path.exists(os.path.join(test_logger_dir, session_name + '.bin')):
            # Check it's not a duplicate
            if logger is not None:
                raise IOError(
                    f'{session_name} found in both {logger} and '
                    f'{test_logger_name}')
            
            # Save logger
            logger = test_logger_name
    
    if logger is None:
        raise IOError(f'cannot find {session_name} in any logger directory')
    
    return logger
    
def convert_dataflow_to_channel_map(dataflow):
    """Convert a dataflow to a channel_map dict suitable for kilosort.
    
    dataflow : DataFrame
        Must have columns: xcoord, ycoord, kcoord
    
    Returns: dict
        n_chan : the number of channels in the binary file
        chanMap : 0-based index of the channels to take
        xc : x-coordinate of each channel in microns
        yc : y-coordinate of each channel in microns
        kcoords : group number of each channel
    """
    ## Create the channel map for kilosort
    # Assume the channels have already been ordered properly in the binary file
    # (that is, in the same order as `dataflow` above)
    # so the channel map for kilosort is simply an arange
    # TODO: figure out how to handle dropped channels. I think in that case,
    # 'n_chan' stays the same, but chanMap0ind is missing some entries
    Nchannels = len(dataflow)
    chanMap0ind = np.arange(Nchannels)
    
    # Extract from dataflow
    xcoords = dataflow['xcoord'].values.astype(float)
    ycoords = dataflow['ycoord'].values.astype(float)
    kcoords = dataflow['kcoord'].values

    # Save the channel map as a matfile in the session directory
    # syntax of these keys has changed for json files vs old matlab files
    # the tolist() is to convert to native python types, not numpy
    ks_json = {
        'n_chan': Nchannels,
        'chanMap': chanMap0ind.tolist(),
        'xc': xcoords.tolist(),
        'yc': ycoords.tolist(),
        'kcoords': kcoords.tolist(),
        }
    
    return ks_json

def load_metadata_from_xml(neural_root, logger, session_name):
    """Load metadata like sampling rate from XML file for a logger recording
    
    Arguments
    ---
    neural_root : path to neural data
    logger : str
        Name of logger, eg logger_628DAB or 628DAB
    session_name : str
        Name of recording, eg 
        'HSW_2025_03_31__17_52_42__09min_58sec__hsamp_128ch_14285sps.bin'
        This is used only to extract the date string by indexing [4:24]
        So it doesn't matter if the ".bin" has been dropped or not
    
    Flow
    ---
    * Form the expected XML file name
    * Raises FileNotFoundError if no XML filename found
    * Load it with xml.etree
    * Parse it into expected datatypes

    Returns: xml_file, xml_data
        xml_file : str, full path to xml file
        xml_data : dict with the following keys
            'binary_filename': str
                Full path to binary file
        
            'sampling_rate_sps' : float, in Hz
                Values around 14285.3 are stored wrong. They are corrected 
                by this script to 32e6 / 2240 = 14285.714285 repeating

            'channel_count : int
                This replaces key 'rec_mode_name' (like '"64ch"') and
                rec_mode (0=64ch, 3=128ch)
            
            'approx_duration_s': int
                This is stored as '0' whenever 'is_still_recording' is True,
                but this function will convert that to np.nan
                
                Otherwise, this will be equal to the difference between
                ts_approx_stop_ack and ts_approx_start_ack.
                
                This is often equal to precise_duration_s, and very often within
                +/- 1 of precise_duration_s. When it is a lot longer than
                precise_duration_s (more than 2 s), likely something went
                wrong. I've never seen it more than 1 s shorter than
                precise_duraton_s.

            'precise_duration_s': int
                This is stored as '0' whenever 'is_still_recording' is True,
                but this function will convert that to np.nan

            'is_still_recording' : bool
                When True, something has likely gone wrong
            
            'stream_duration_s' : int
                This one will be 0 if 'is_still_recording' is True
            
            'sync_frequency' : float
                1e6 if this key was "1.000 MHz"
                100e3 if this key was "100.000 kHz"
                np.nan if the key was "0.0"
                The redundant key "sync_enabled" is checked to be 0 when 
                this value is np.nan, and 1 otherwise, and then dropped
                
            'version' : str
            
            'ts_approx_start_ack' : datetime
                Parsed from "YYYY.MM.DD HH:MM:SS"
            
            'ts_approx_stop_ack' : datetime
                Parsed from "YYYY.MM.DD HH:MM:SS"
                This is stored as '0' whenever 'is_still_recording' is True,
                but this function will convert that to np.nan
            
            'ts_approx_start_cmd' : int, datetime
                Parsed from seconds from epoch
        
        The key "session_type" is always "recording", so it is checked and
        dropped.
        The key "valid", referring to headstage stack, is always True, 
        so it is checked and dropped.
    """
    
    ## Find the corresponding xml file
    # Get date string from session name
    date_string = session_name[4:24]
    
    # Normalize logger
    logger = logger.replace('logger_', '')
    
    # Join 
    xml_file = os.path.join(
        neural_root, f'record_ID{logger}_{date_string}.xml')
    
    # Check that it exists
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"xml file does not exist at {xml_file}")

    
    ## Parse xml
    # Parse with xml module
    try:
        tree = xml.etree.ElementTree.parse(xml_file)
    except xml.etree.ElementTree.ParseError:
        raise IOError(f"cannot parse xml at {xml_file}")

    # Helper function
    def helper_get(tree, path):
        nodes = tree.findall(path)
        assert len(nodes) == 1
        return nodes[0].text

    # Parse
    xml_data = {}
    xml_data['rec_mode_name'] = (
        helper_get(tree, 'RECORD/REC_MODE_NAME'))
    xml_data['sampling_rate_sps'] = float(
        helper_get(tree, 'SETTINGS/SAMPLING_RATE_SPS'))
    xml_data['rec_mode'] = (
        helper_get(tree, 'RECORD/REC_MODE'))
    xml_data['approx_duration_s'] = float(
        helper_get(tree, 'RECORD/APPROX_DURATION_S'))
    xml_data['stream_duration_s'] = float(
        helper_get(tree, 'RECORD/STREAM_DURATION_S'))
    xml_data['precise_duration_s'] = float(
        helper_get(tree, 'RECORD/PRECISE_DURATION_S'))
    xml_data['sync_enabled'] = (
        helper_get(tree, 'SYNCHRONISATION/SYNC_ENABLED'))
    xml_data['sync_frequency'] = (
        helper_get(tree, 'SYNCHRONISATION/SYNC_FREQUENCY'))
    xml_data['session_type'] = (
        helper_get(tree, 'RECORD/SESSION_TYPE'))
    xml_data['is_still_recording'] = (
        helper_get(tree, 'RECORD/IS_STILL_RECORDING'))
    xml_data['version'] = (
        helper_get(tree, 'INFO/VERSION'))
    xml_data['valid'] = (
        helper_get(tree, 'HEADSTAGE/VALID'))
    xml_data['ts_approx_start_cmd'] = (
        helper_get(tree, 'RECORD/TS_APPROX_START_CMD'))
    xml_data['ts_approx_start_ack'] = (
        helper_get(tree, 'RECORD/TS_APPROX_START_ACK'))
    xml_data['ts_approx_stop_ack'] = (
        helper_get(tree, 'RECORD/TS_APPROX_STOP_ACK'))

    
    ## Additional parsing
    # Ensure these strs are ints
    for key in [
            'ts_approx_start_cmd', 
            'approx_duration_s', 
            'stream_duration_s', 
            'precise_duration_s',
            'sync_enabled',
            ]:
        assert int(xml_data[key]) == float(xml_data[key])
        xml_data[key] = int(xml_data[key])

    # Ensure these strs are bools
    for key in [
            'valid', 
            'is_still_recording', 
            ]:
        if xml_data[key] == 'true':
            xml_data[key] = True
        
        elif xml_data[key] == 'false':
            xml_data[key] = False
        
        else:
            raise ValueError(
                f'invalid boolean value {xml_data[key]} for key {key}')
    
    # Set channel count, and pop redundant rec_mode and rec_mode_name
    xml_data['channel_count'] = int(
        xml_data['rec_mode_name'].replace('"', '').replace('ch', ''))
    if xml_data['channel_count'] == 64:
        assert xml_data['rec_mode_name'] == '"64ch"'
        assert xml_data['rec_mode'] == '0'
        xml_data.pop('rec_mode_name')
        xml_data.pop('rec_mode')
    
    elif xml_data['channel_count'] == 128:
        assert xml_data['rec_mode_name'] == '"128ch"'
        assert xml_data['rec_mode'] == '3'
        xml_data.pop('rec_mode_name')
        xml_data.pop('rec_mode')
    
    else:
        raise ValueError("unsupported channel count: {}".format(
            xml_data['channel_count']))
    
    # Convert this integer timestamp to datetime
    xml_data['ts_approx_start_cmd'] = datetime.datetime.fromtimestamp(
        xml_data['ts_approx_start_cmd'])
    
    # Convert these string timestamps to datetime
    xml_data['ts_approx_start_ack'] = datetime.datetime.strptime(
        xml_data['ts_approx_start_ack'], '%Y.%m.%d %H:%M:%S')
    
    # Deal with is_still_recording
    if not xml_data['is_still_recording']:
        # The usual case
        xml_data['ts_approx_stop_ack'] = datetime.datetime.strptime(
            xml_data['ts_approx_stop_ack'], '%Y.%m.%d %H:%M:%S')    
    else:
        # The broken case
        xml_data['ts_approx_stop_ack'] = np.nan
        
        # This will be set to zero, but is actually null
        assert xml_data['approx_duration_s'] == 0
        xml_data['approx_duration_s'] = np.nan

        # This will be set to zero, but is actually null
        assert xml_data['precise_duration_s'] == 0
        xml_data['precise_duration_s'] = np.nan

    # Drop this useless one
    assert xml_data['session_type'] == 'recording'
    xml_data.pop('session_type')
    
    # Drop this useless one
    assert xml_data['valid'] == True
    xml_data.pop('valid')
    
    # Convert this one to more meaningful units
    if xml_data['sync_frequency'] == '0.0':
        xml_data['sync_frequency'] = np.nan
        assert xml_data['sync_enabled'] == 0
    
    elif xml_data['sync_frequency'] == '100.000 kHz':
        xml_data['sync_frequency'] = 100e3
        assert xml_data['sync_enabled'] == 1
    
    elif xml_data['sync_frequency'] == '1.000 MHz':
        xml_data['sync_frequency'] = 1e6
        assert xml_data['sync_enabled'] == 1
    
    else:
        raise ValueError(
            f"invalid sync_frequency: {xml_data['sync_frequency']}")

    # Drop this now useless one
    xml_data.pop('sync_enabled')
    

    ## Correct 14285 which is stored wrong
    if (
            (xml_data['sampling_rate_sps'] > 14285) and 
            (xml_data['sampling_rate_sps'] < 14286)
        ):
        
        # The correct value is 14285.714285 repeating
        xml_data['sampling_rate_sps'] = 32e6 / 2240


    ## Additionally add metadata about the binary file
    xml_data['binary_filename'] = os.path.join(
        neural_root, f'logger_{logger}', session_name + '.bin')

    # Calculate size of file in bytes
    packed_file_size_bytes = os.path.getsize(xml_data['binary_filename'])
    
    # Calculate length of recording in samples
    offset = 8
    packed_file_size_samples = (
        (packed_file_size_bytes - offset) // xml_data['channel_count'] // 2)
    
    # Check file size makes sense
    expected_size = (
        packed_file_size_samples * xml_data['channel_count'] * 2 + offset)
    if expected_size != packed_file_size_bytes:
        raise IOError(
            f'{xml_data["binary_filename"]} was {packed_file_size_bytes} '
            'but it should have been '
            f'{expected_size} bytes')
    
    # Convert file size to time
    xml_data['file_duration_s'] = (
        packed_file_size_samples / xml_data['sampling_rate_sps'])
    
    
    ## Return
    return xml_file, xml_data

def form_analog_filename(analog_root, analog_session, experiment_number=1, 
    recording_number=1):
    """Form the full path to the analog file
    
    This function contains the defaults that usually work for analog data
    collected by the eCube. 
    
    analog_root: path
        Should end in 'd_drive'
    
    analog_session: str
        Should correspond to a session name within analog_root
    
    experiment_number, recording_number : int
        These are usually 1 but can be other numbers depending on how you
        clicked play and record in OpenEphys
    
    Returns : analog_packed_filename
        A full path to continuous.dat file
    """
    analog_packed_filename = os.path.join(
        analog_root, 
        analog_session, 
        'Record Node 107', 
        f'experiment{experiment_number}', 
        f'recording{recording_number}', 
        'continuous',
        'eCube_Server-105.0', 
        'continuous.dat',
        )    
    
    return analog_packed_filename

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

def load_neural_data(neural_packed_filename, n_channels, offset=8):
    """Memmap packed neural data
    
    neural_packed_filename : str
        Path to neural data
    
    n_channels : int
    
    offset : int
        This is 8 for White Matter data
    
    Returns : memmap
        It will have `n_channels` columns
    """
    # Calculate size of file in bytes
    packed_file_size_bytes = os.path.getsize(neural_packed_filename)
    
    # Calculate length of recording in samples
    packed_file_size_samples = (
        (packed_file_size_bytes - offset) // n_channels // 2)
    
    # Check file size makes sense
    expected_size = packed_file_size_samples * n_channels * 2 + offset
    if expected_size != packed_file_size_bytes:
        raise ValueError(
            f'file was {packed_file_size_bytes} but it should have been '
            f'{expected_size} bytes')
    
    # Memmap
    neural_mm = np.memmap(
        neural_packed_filename, 
        np.int16, 
        'r', 
        offset=offset,
        shape=(packed_file_size_samples, n_channels)
    )   
    
    return neural_mm

def load_open_ephys_data(directory, recording_idx=0, convert_to_microvolts=True):
    """Load OpenEphys data
    
    For debugging (i.e., determining how many recording_idx there are, try
        session = open_ephys.analysis.Session(directory)
    I think len(session.recordnodes) is always 1 for our setup
    sesion.recordnodes.recordings is a list of recordings. The distinction
    between experiments and recordings is ignored, it's just a simple list.
    
    TODO: how do you get the start time of each recording?
    
    directory : path to session
    recording_idx : int, which recording to get
    convert_to_microvolts : bool
        If True, neural_data and analog_data are converted to microvolts
        and volts, respectively
        If False, they are left as memmap (in bit-levels)
    
    Returns: dict
        'metadata': includes conversion factor bit_volts
        'timestamps': memmap of timestamps in seconds
        'neural_data': all neural data (columns are channels)
        'analog_data': all analog data (columns are channels)
    """
    # Hide this import here because it's uncommon
    import open_ephys.analysis
    
    # Form session
    session = open_ephys.analysis.Session(directory)
    n_recordings = len(session.recordnodes[0].recordings)

    # Extract data
    metadata = session.recordnodes[0].recordings[
        recording_idx].continuous[0].metadata
    timestamps = session.recordnodes[0].recordings[
        recording_idx].continuous[0].timestamps
    data = session.recordnodes[0].recordings[
        recording_idx].continuous[0].samples

    # Split into neural and analog
    # The channel names are in metadata['channel_names']
    neural_data = data[:, :-8]
    analog_data = data[:, -8:]

    # Convert to array and get into physical units
    if convert_to_microvolts:
        neural_data = neural_data * metadata['bit_volts'][0]
        analog_data = analog_data * metadata['bit_volts'][-1]
    
    # Return
    return {
        'metadata': metadata,
        'timestamps': timestamps,
        'neural_data': neural_data,
        'analog_data': analog_data,
        }

def get_video_start_time(*args, **kwargs):
    # only for deprecations below
    import paclab.syncing

    print(
        'warning: replace all calls to paclab.neural.get_video_start_time '
        'with paclab.syncing.get_video_start_time instead'
        )
    return paclab.syncing.get_video_start_time(*args, **kwargs)

def get_recording_start_time(*args, **kwargs):
    # only for deprecations below
    import paclab.syncing

    print(
        'warning: replace all calls to paclab.neural.get_recording_start_time '
        'with paclab.syncing.get_recording_start_time instead'
        )
    return paclab.syncing.get_recording_start_time(*args, **kwargs)

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

def load_spike_clusters(sort_dir):
    """Load the cluster of each spike from kilosort data
    
    This includes any reclustering that was done in phy
    """
    spike_cluster = np.load(os.path.join(sort_dir, 'spike_clusters.npy'))
    return spike_cluster

def load_spikes(sort_dir):
    """Load spike times from kilosort
    
    This is just the data in spike_times.npy, flattened
    Data is converted to int (in case it is stored as uint64)
    
    Returns: 
        spike_time_samples
    """
    spike_time_samples = np.load(
        os.path.join(sort_dir, 'spike_times.npy')).flatten().astype(int)
    
    return spike_time_samples

def load_spike_templates1(sort_dir):
    """Return spike templates from kilosort

    These are the actual templates that were used, not the templates
    for each spike. For that, use load_spike_templates2
    
    Returns: templates
        array with shape (n_templates, n_timepoints, n_channels)
    """
    templates = np.load(os.path.join(sort_dir, 'templates.npy'))
    return templates

def load_spike_amplitudes(sort_dir):
    """Return spike amplitudes from kilosort
    
    """
    # Amplitude of every spike
    spike_amplitude = np.load(os.path.join(sort_dir, 'amplitudes.npy'))
    
    return spike_amplitude.flatten()

def load_cluster_groups(sort_dir):
    """Returns type (good, MUA, noise) of each cluster from kilosort"""
    # This has n_manual_clusters rows, with the group for each
    cluster_group = pandas.read_table(os.path.join(sort_dir, 
        'cluster_group.tsv'))
    
    return cluster_group
