# TODO: Remove functions we don't need, move any functions we do need into
# dedicated files here with meaningful names (loading, signal_processing, etc)

import math
import os.path
import struct
import numpy as np
import scipy
import pickle
import pandas
import socket
import matplotlib
import matplotlib.pyplot as plt
import my.plot

def clear_all_lines_and_images_from_figure(fig):
    """Remove all `lines` and `images` from all axes in `fig`"""
    for ax in fig.axes:
        clear_all_lines_and_images(ax)

def clear_all_lines_and_images(ax):
    """Remove all `lines` and `images` from `ax`"""
    while len(ax.lines) > 0:
        ax.lines[0].remove()

    while len(ax.images) > 0:
        ax.images[0].remove()

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def get_metadata(notes_directory, data_directory, datestring,
                 metadata_version, day_directory = '_ABR'):
    """Get the metadata from a day of recordings

    Looks for the metadata csv file
    Reads this csv file
    Includes only the rows where include is true
    Forms a session_name column
    Forms a datafile column

    Parameters:
        notes_directory: where the metadata csv
        data_directory,
        datestring: yymmdd datestring
        metadata_verbose: using the verbose format with 2 extra columns for resistor values
        day_directory: what the directory's name is, defaulting to '{date}_ABR'
    Returns: DataFrame
    """
    # Rowan started organizing ABR_data by year and right now the only year that isn't in its own subfolder is 2024
    if datestring[0:2] != '24':
        notes_directory = os.path.join(notes_directory, '20' + datestring[0:2])

    # Rowan also changed their naming convention for folder names from '20yymmdd' to just 'yymmdd'
    if datestring[0:2] != '25':
        # Form the filename to the csv file
        if metadata_version == "v4":
            csv_filename = os.path.join(
                notes_directory,
                '20' + datestring + day_directory,
                datestring + '_notes_v4.csv')
        elif metadata_version == "verbose":
            csv_filename = os.path.join(
                notes_directory,
                '20' + datestring + day_directory,
                datestring + '_notes_verbose.csv')
        else:
            csv_filename = os.path.join(
                notes_directory,
                '20' + datestring + day_directory,
                datestring + '_notes.csv')
    # There's no reason for a 2025 recording to have the legacy notes_verbose format so skip that
    elif datestring[0:2] == '25':
        # Form the filename to the csv file
        if metadata_version == "v4":
            csv_filename = os.path.join(
                notes_directory,
                datestring + day_directory,
                datestring + '_notes_v4.csv')
        else:
            csv_filename = os.path.join(
                notes_directory,
                datestring + day_directory,
                datestring + '_notes.csv')

    # Read the CSV file
    metadata = pandas.read_csv(csv_filename)
    metadata['include'] = metadata['include'].fillna(1).astype(bool)

    # Drop the ones that we don't want to include
    # metadata = metadata.loc[metadata['include'].values]

    # Form session name from session number
    metadata['session_name'] = [
        'BG-{:04d}'.format(n) for n in metadata['session'].values]

    # Form the full path to the session
    metadata['datafile'] = [
        os.path.join(data_directory, session_name + '.bin')
        for session_name in metadata['session_name']]

    return metadata

def parse_header(header):
    """Parse data from a single header

    `header`: 60 bytes

    Returns: dict, with the following items:
        packet_number : number of packet within the file
        number_channels : typically 8
        number_samples : this is the number of samples per packet
        trigger_position : whether a trigger occurred in this packet
            -1 if no trigger
            Otherwise, an index into the packet
        header_ver : typically 2
        total_pkts : total number of packets in the file
            This is only valid if it's the first header and if Labview
            completed the recording. Otherwise it's zero.
        sampling_sps : sampling rate
        gains : an array of channel gains
    """
    res = {}

    res['packet_number'] = struct.unpack('<I', header[0:4])[0]
    res['number_channels'] = struct.unpack('<I', header[4:8])[0]
    res['number_samples'] = struct.unpack('<I', header[8:12])[0]
    res['trigger_position'] = struct.unpack('<i', header[12:16])[0]
    res['header_ver'] = struct.unpack('<i', header[16:20])[0]
    res['total_pkts'] = struct.unpack('<I', header[20:24])[0]
    res['sampling_sps'] = struct.unpack('<i', header[24:28])[0]
    res['gains'] = np.array(struct.unpack('<8i', header[28:60]))

    return res

def parse_data(data_bytes, number_channels, number_samples):
    """Parse bytes into acquired data

    Takes the `data_bytes` that have been read from disk, removes all
    the header information, converts to float, concatenates across packets,
    and returns.

    data_bytes : bytes
        This is the data read from disk

    Returns: array of shape (n_samples, n_channels)
        This is all the data from each channel.
    """
    # Each packet comprises a 60 byte headers (15 ints) and a 16000 byte data
    # payload (4000 floats)

    # It's easiest to convert everything to float, and then just dump the
    # floats that correspond to the headers

    # Unpack it all to floats
    n_floats = len(data_bytes) // 4

    # The old, slow way
    # data = np.array(struct.unpack('<{}f'.format(n_floats), data_bytes))

    # This is faster
    # https://stackoverflow.com/questions/36797088/speed-up-pythons-struct-unpack
    data = np.ndarray(
        (n_floats,), dtype=np.float32, buffer=data_bytes).astype(float)

    # Truncate to a multiple of 4015
    total_packet_size = 15 + number_channels * number_samples
    n_complete_packets = len(data) // total_packet_size
    if len(data) / total_packet_size != n_complete_packets:
        print("warning: data length was not a multiple of packet length")
        data = data[:total_packet_size * n_complete_packets]

    # Reshape into packets -- one packet (4015 bytes) per row
    data = data.reshape(-1, total_packet_size)

    # Drop the first 15 bytes of each row, which are the headers
    # TODO: instead of dropping, convert to ints, and extract header
    # of each packet
    data = data[:, 15:]

    # Each row needs to be reshaped into (number_channels, number_samples)
    # Note that the data is structured as sample-first;channel-second
    # which doesn't really make sense
    data = data.reshape((-1, number_channels, number_samples))

    # Concatenate all packets together (axis=1 accounts for the sample-first
    # problem). Transpose so each channel is a column
    data = np.concatenate(data, axis=1).T

    return data

def extract_single_ad_and_nrl(data,neural_channel, speaker_channel, audio_drop_threshold, neural_drop_threshold):
    """
    Takes a single dession of parsed data and extracts just the audio and neural channels.
    Drops munged data that's outside the drop thresholds.

    Args:
        data: output of parse_data()
        neural_channel: channel of neural data in the binary file
        speaker_channel: channel of audio data in the binary file
        audio_drop_threshold: drop outliers/munged if abs(voltage) is above this
        neural_drop_threshold: drop outliers/munged if abs(voltage) is above this

    Returns:
        audio_data: ndarray of the session's audio voltages in V
        neural_data: ndarray of the session's neural voltages in uV

    """
    # Extract audio data
    audio_data = data[:, speaker_channel]
    neural_data = data[:, neural_channel]

    ## Find and drop munged data where voltage is way outside thresholds
    bad_mask = np.append(
        np.where(np.abs(audio_data) > audio_drop_threshold)[0],
        np.where(np.abs(neural_data) > neural_drop_threshold)[0])
    bad_mask = np.sort(bad_mask)

    # Drop bad_mask indexes from data
    # If there are two bad areas in the data, this will drop everything
    # in between those areas too
    # TODO: reimplement with my.misc.times_near_times
    if len(bad_mask) != 0:
        if bad_mask[-1] >= len(audio_data) + 4:
            audio_data = audio_data[:bad_mask[0] - 5]
            neural_data = neural_data[:bad_mask[0] - 5]
        else:
            audio_data = np.concatenate((
                audio_data[:bad_mask[0] - 5], audio_data[bad_mask[-1] + 5:]))
            neural_data = np.concatenate((
                neural_data[:bad_mask[0] - 5], neural_data[bad_mask[-1] + 5:]))

    ## de-median audio data
    audio_data = audio_data - np.median(audio_data)

    # Convert neural data to microvolts
    # neural_data_uV = neural_data*1e6

    return audio_data,neural_data

def find_extrema(flat_data, pk_threshold, wlen=None, diff_threshold = None, distance = 200, width = None,plateau_size=None):
    if type(flat_data) == np.ndarray:
        flat_data = pandas.Series(flat_data)
    pos_peaks = scipy.signal.find_peaks(
        flat_data,wlen = wlen,threshold=diff_threshold,
        height=pk_threshold, distance=distance,
        width=width, plateau_size=plateau_size)[0]
    neg_peaks = scipy.signal.find_peaks(
        -flat_data, wlen = wlen, threshold=diff_threshold,
        height=pk_threshold, distance=distance,
        width=width, plateau_size=plateau_size)[0]
    return pos_peaks,neg_peaks
def get_peak_ys(peak_xs,original_data):
    peak_ys = []
    for x in peak_xs:
        peak_ys.append(pandas.Series(original_data).iloc[x])
    return peak_ys
def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]
def get_single_onsets(audio_data,audio_threshold, abr_start_sample = -80, abr_stop_sample = 120,distance=200):
    pos_peaks, neg_peaks = find_extrema(audio_data,audio_threshold,distance=distance)
    # Concatenate pos and neg peaks,
    #  then drop ringing/overshoot peaks during refractory period
    onsets = np.sort(np.concatenate([pos_peaks, neg_peaks]))

    # Debugging plot of audio and onsets
    # onset_ys = get_peak_ys(onsets, audio_data)
    # plt.plot(audio_data)
    # plt.plot(onsets, onset_ys, "x")

    onsets2 = drop_refrac(onsets, 750)
    # Get rid of onsets that are too close to the start or end of the session.
#    if onsets2[0] < 80: onsets2 = onsets2[1:]
    onsets2 = onsets2[
        (onsets2 > -abr_start_sample) &
        (onsets2 < len(audio_data) - abr_stop_sample)
        ]
    return onsets2

def get_data_without_onsets(metadata, datestring, header_size, neural_channel, speaker_channel,
        audio_drop_threshold,neural_drop_threshold, day_directory = "_ABR",
        has_extra_channel = False, extra_channel = 2):
    """Parses all of the LV binaries from a certain date
    Removes sections with impossibly high/low values but doesn't extract audio onsets
    Does de-median the audio data at the end

    Arguments
        metadata_verbose: T/F, whether or not you're using the verbose format
        datestring: 6 character string for experiment date
        header_size: size of header
        neural_channel: channel of neural data in the binary file
        speaker_channel: channel of audio data in the binary file
    Returns
        audio_l: list
            The list is the same length as `metadata`, unless broken files
            were skipped. Each item in the list is an array of audio data
            from the corresponding session.
        neural_l: list
            Analogous to `audio_data`
            Each item in this list is the same length as the corresponding
            item in `audio_data`
        metadata: dataframe
            Metadata imported from the csv notes file.
        sampling_rate: int
            Samples per second, a setting on the ADS1299. Useful for downstream analysis.
    """

    ## Iterate over rows in metadata
    audio_l = []
    neural_l = []
    extrach_l = []
    for metadata_idx in metadata.index:

        ## Get the name of the data file
        datafile = metadata.loc[metadata_idx, 'datafile']
        session_name = metadata.loc[metadata_idx, 'session_name']
        include = metadata.loc[metadata_idx,'include']
        print("loading {}".format(datafile))

        # Skip if it doesn't exist
        if not os.path.exists(datafile):
            print("warning: {} does not exist".format(datafile))
            continue


        ## Open the file and read the header
        # Read the data
        with open(datafile, "rb") as fi:
            data_bytes = fi.read()

        # Skip if nothing
        if len(data_bytes) == 0:
            print("warning: {} is empty".format(datafile))
            sampling_rate = "skipped"
            continue

        # We need to parse the first header separately because it has data that
        # we need to parse the rest of the packets.
        first_header_bytes = data_bytes[:header_size]
        first_header_info = parse_header(first_header_bytes)

        # We use this a lot so extract from the dict
        sampling_rate = first_header_info['sampling_sps']


        ## Parse the entire file
        data = parse_data(
            data_bytes,
            first_header_info['number_channels'],
            first_header_info['number_samples'])
        ## Extract audio and neural data
        audio_data, neural_data = extract_single_ad_and_nrl(
            data,neural_channel,speaker_channel,
            audio_drop_threshold,neural_drop_threshold)
        if has_extra_channel:
            extrach_data = data[:, extra_channel]
        ## Store
        audio_l.append([session_name,'audio', include, audio_data])
        neural_l.append([session_name,'neural',include, neural_data])
        if has_extra_channel:
            extrach_l.append([session_name, 'extra',include, extrach_data ])
    if has_extra_channel == False:
        extrach_l = ['No data']
    return audio_l, neural_l, extrach_l, sampling_rate

def align_singleday_data(audio_data, neural_data, extrach_data, metadata,
     audio_threshold, sampling_rate=16000,
     abr_start_sample=-80, abr_stop_sample = 120,
     has_extra_channel = False, distance = 200):
    """Take audio, neural, and metadata from get_data_without_onsets.

    Step 1: Find audio onsets and align data to them
    Step 2: Distinguish the characteristics of the audio pulse
        Positive/Negative and Loud/Quiet
    Step 3: Output a dataframe for each channel (audio and neural)

    Args:
        audio_data: list of ndarrays
        List of arrays where each array is the raw speaker voltages from
        a single recording session

        neural_data: list of ndarrays
        List of arrays where each array is the raw speaker voltages from
        a single recording session

        metadata: pandas.df
        Dataframe from get_verbose_metadata

        sampling_rate: int
        ADS1299 sampling rate for that session, should be 16,000 samples/s

    Returns:
        session_ad: df of a single day's parsed and aligned speaker voltages
        session_neural:df of a single day's parsed and aligned neural voltages

    """
    ## Iterate over rows of metadata
    triggered_ad_l = []
    triggered_neural_l = []
    triggered_extrach_l = []
    keys_l = []

    for metadata_idx in metadata.index:
        session_name = metadata.loc[metadata_idx, 'session_name']

        onsets, pulse_direction = get_click_info(
            audio_data, metadata,metadata_idx, audio_threshold, abr_start_sample, abr_stop_sample,distance=distance)

        ## Align audio and neural data around triggers (onsets)
        # Audio
        triggered_ad = np.array([
            audio_data[metadata_idx][3][trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])

        # Neural
        triggered_neural = np.array([
            neural_data[metadata_idx][3][trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])

        #Extra channel
        if has_extra_channel:
            triggered_extrach = np.array([
                extrach_data[metadata_idx][3][trigger + abr_start_sample:trigger + abr_stop_sample]
                for trigger in onsets])
        else:
            triggered_extrach = ["No extra channel"]
        ## Dataframe the results
        # Define time course in ms
        t_plot = np.arange(
            abr_start_sample, abr_stop_sample) / sampling_rate * 1000

        # Create DataFrame with t_plot on the columns, and multi-index
        # with stimulus info on the rows
        tad_mindex = pandas.MultiIndex.from_arrays(
            [pulse_direction, np.arange(0, len(triggered_ad))],
            names=('ad_pulse_V', 'trial'))

        # Audio
        triggered_ad_df = pandas.DataFrame(
            triggered_ad, index=tad_mindex, columns=t_plot)
        triggered_ad_df.columns.name = 'timepoint'

        # Neural
        triggered_neural_df = pandas.DataFrame(
            triggered_neural, index=tad_mindex, columns=t_plot)
        triggered_neural_df.columns.name = 'timepoint'

        # Extra channel
        if has_extra_channel:
            triggered_extrach_df = pandas.DataFrame(
                triggered_extrach, index=tad_mindex, columns=t_plot)
            triggered_extrach_df.columns.name = 'timepoint'
        else:
            triggered_extrach_df = ["No extra channel"]

        ## Store
        triggered_ad_l.append(triggered_ad_df)
        triggered_neural_l.append(triggered_neural_df)
        triggered_extrach_l.append(triggered_extrach_df)
        keys_l.append(session_name)

    ## Concat
    session_ad = pandas.concat(
        triggered_ad_l, keys=keys_l, names=['session'])
    session_neural = pandas.concat(
        triggered_neural_l, keys=keys_l, names=['session'])
    if has_extra_channel:
        session_extrach = pandas.concat(
            triggered_extrach_l, keys=keys_l, names=['session'])
    else:
        session_extrach = ["No extra channel"]
    return session_ad, session_neural,session_extrach

def get_click_info(audio_data, metadata, metadata_idx, audio_threshold, abr_start_sample, abr_stop_sample,distance=200):

    session_name = metadata.loc[metadata_idx, 'session_name']

  # Get onsets
    onsets = get_single_onsets(
        audio_data[metadata_idx][3],audio_threshold,abr_start_sample,abr_stop_sample)

    # Debugging plot of audio and onsets
    # onset_ys = get_peak_ys(onsets, audio_data[metadata_idx][3])
    # plt.plot(audio_data[metadata_idx][3])
    # plt.plot(onsets,onset_ys,"x")

    # Detect if the audio pulse was positive or negative V
    # @RG TODO: replace this block of code with
    # RG: THIS DOESN'T WORK I don't know why but it doesn't give you the same result
    # pulse_direction2 = audio_data[metadata_idx][3][onsets] >= 0
    pulse_direction = []
    pulse_amplitude=[]
    for audio_pulse in np.arange(0, len(onsets)):
        # Determine if the audio pulse was positive or negative V
        # Average the detected onset's voltage with the surrounding 2 voltages just to make sure
        # @RG: should not be necessary to average because onsets2 indexes
        # peak, right?
        # @CR: Sure it's probably not necessary but it doesn't hurt
        pulsewindow_avg = audio_data[metadata_idx][3][onsets[audio_pulse] - 1:onsets[audio_pulse] + 2].mean()
        if pulsewindow_avg >= 0:
            pulse_direction.append('positive')
        if pulsewindow_avg < 0:
            pulse_direction.append('negative')
    return onsets, pulse_direction


def get_click_info_df(audio_data, metadata, audio_threshold, abr_start_sample, abr_stop_sample):
    click_info_l = []
    sessions_l = []
    for metadata_idx in metadata.index:
        session_name = metadata.loc[metadata_idx, 'session_name']

      # Get onsets
        onsets = get_single_onsets(
            audio_data[metadata_idx][3],audio_threshold,abr_start_sample,abr_stop_sample)

        # Debugging plot of audio and onsets
        # onset_ys = get_peak_ys(onsets, audio_data[metadata_idx][3])
        # plt.plot(audio_data[metadata_idx][3])
        # plt.plot(onsets,onset_ys,"x")

        # Detect if the audio pulse was positive or negative V
        # @RG TODO: replace this block of code with
        # RG: THIS DOESN'T WORK I don't know why but it doesn't give you the same result
        # pulse_direction2 = audio_data[metadata_idx][3][onsets] >= 0
        pulse_direction = []
        pulse_amplitude=[]
        for audio_pulse in np.arange(0, len(onsets)):
            # Determine if the audio pulse was positive or negative V
            # Average the detected onset's voltage with the surrounding 2 voltages just to make sure
            # @RG: should not be necessary to average because onsets2 indexes
            # peak, right?
            # @CR: Sure it's probably not necessary but it doesn't hurt
            pulsewindow_avg = audio_data[metadata_idx][3][onsets[audio_pulse] - 1:onsets[audio_pulse] + 2].mean()
            if pulsewindow_avg >= 0:
                pulse_direction.append('positive')
            if pulsewindow_avg < 0:
                pulse_direction.append('negative')
        for onset_idx in range(0,len(onsets)):
            click_info_l.append([session_name,onsets[onset_idx],pulse_direction[onset_idx]])
    click_info_df = pandas.DataFrame(click_info_l, columns=["session_name", "onset", "pulse_direction"])
    return click_info_df


def join_column_to_index(df1, df2, columns, on=None, reorder_levels=None,
                         sort=True):
    """Join df2[columns] onto the index of df1

    df1 : DataFrame to join onto
    df2 : DataFrame to take columns from
    columns : string or list of strings
        Either that column or that list of columns will be taken from df2
        and joined on df1
    on : passed to df1.join. If None it will use default
    reorder_levels : list or None
        If not None, reorder the levels of the MultiIndex of df1 after joining
        This must be a complete list of all levels, both the originals and
        the newly joined
    sort : bool, default True
        Whether to sort the result by its new index before returning

    See here:
    https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex/56278736#56278736

    Returns: DataFrame
        df1 with df2[columns] joined onto it
    """
    df1_old = df1.droplevel([2, 3], axis=0)
    df2_old = df2
    # Get idx as frame
    old_idx = df1.index.to_frame()
    old_idx2= df1_old.index.to_frame()

    # Join
    # Rename session_name to session so that it matches old_idx
    to_join = df2.reset_index().drop('original_session_num', axis=1)

    # Reindex by date and session to match old_idx
    to_join = to_join.set_index(['date', 'session'])

    # Keep only the columns that we want to join
    to_join = to_join[columns]

    # Join onto old_idx
    new_idx = old_idx.join(to_join)

    # Convert back to multiindex
    res = df1.copy()
    res.index = pandas.MultiIndex.from_frame(new_idx)

    # Reorder levels
    if reorder_levels is not None:
        res = res.reorder_levels(reorder_levels)

    # Sort
    if sort:
        res = res.sort_index()

    # Return
    return res

def get_audio_amplitude(big_audio_data,big_neural_data,amplitude_thresholds):
    return

def single_row_extrema(flat_data,ms_l):
    # Almost the same as find_extrema, except
    # it doesn't require a threshold and it keeps the xs and ys seperate
    if type(flat_data) == np.ndarray:
        flat_data = pandas.Series(flat_data)
    find_peaks = scipy.signal.find_peaks(flat_data)[0]
    peak_xs = []
    peak_ys = []
    for i in find_peaks:
        peak_xs.append(flat_data.index[i])
        peak_ys.append(flat_data.iloc[i])
    flat_data_np = np.array(flat_data)
    find_mins = scipy.signal.argrelmin(flat_data_np)[0]
    valley_xs = []
    valley_ys = []
    for i in find_mins:
        valley_xs.append(flat_data.index[i])
        valley_ys.append(flat_data.iloc[i])

    return peak_xs,peak_ys,valley_xs,valley_ys
def get_peak_coords_df(channel, ch_avgs,ms_l):
    if channel==1:
        ch_avgs = ch_avgs.loc['neural']
        ch_avgs = ch_avgs.groupby(['date', 'mouse','amplitude', 'ch1_config', 'speaker_side']).mean()
        ch_avgs.index = ch_avgs.index.rename('ch_config',level='ch1_config')
    if channel == 3:
        ch_avgs = ch_avgs.loc['extrach']
        ch_avgs = ch_avgs.groupby(['date', 'mouse','amplitude', 'ch3_config', 'speaker_side']).mean()
        ch_avgs.index = ch_avgs.index.rename('ch_config',level='ch3_config')
    ABR_stats = []
    for ABRidx in np.arange(0, len(ch_avgs)):
        if ABRidx == len(ch_avgs):
            singleABR = ch_avgs[ABRidx:]
        singleABR = ch_avgs[ABRidx:ABRidx + 1]

        pospk_xs, pospk_ys, negpk_xs, negpk_ys = single_row_extrema(singleABR.values[0], ms_l)
        all_pk_xs = np.concatenate([pospk_xs, negpk_xs])
        all_pk_ys = get_peak_ys(all_pk_xs, np.array(singleABR)[0])

        pk_direction = []
        for yval in all_pk_ys:
            if yval >= 0:
                pk_direction.append('pospk')
            elif yval < 0:
                pk_direction.append('negpk')
            else:
                pk_direction.append('ERROR')

        d = {
            # "info":         singleABR.index,
            "pk_x": all_pk_xs.astype(int),
            "pk_y": all_pk_ys,
            "pk_direction": pk_direction
        }
        pk_info = pandas.DataFrame(d)
        pk_info['speaker_side'] = singleABR.index.get_level_values('speaker_side').values[0]
        pk_info['amplitude'] = singleABR.index.get_level_values('amplitude').values[0]
        pk_info['ch_config'] = singleABR.index.get_level_values('ch_config').values[0]
        pk_info['mouse'] = singleABR.index.get_level_values('mouse').values[0]
        pk_info['date'] = singleABR.index.get_level_values('date').values[0]
#        pk_info['data'] = singleABR.index.get_level_values('data').values[0]
        ABR_stats.append(pk_info)
    all_pks_df = pandas.concat(ABR_stats).sort_values(['date', 'mouse', 'amplitude', 'ch_config', 'speaker_side', 'pk_x']).reset_index(
        drop=True)
    all_pks_df = all_pks_df.set_index(['date', 'mouse', 'amplitude', 'ch_config', 'speaker_side']).sort_index()
    return all_pks_df
def get_singleday_peak_coords_df(channel, ch_avgs,ms_l):
    # Channels called as "special1" or "special3" are to catch a situation where
    # you recorded vertex-ear on channel 1 instead of channel 3, or LR on ch 3 instead of ch 1
    if channel == "special1":
        ch_avgs.index = ch_avgs.index.rename('ch_config',level='ch1_config')
    if channel == "special3":
        ch_avgs.index = ch_avgs.index.rename('ch_config', level='ch3_config')
    if channel==1:
        ch_avgs = ch_avgs.loc['neural']
        ch_avgs = ch_avgs.groupby(['mouse','amplitude', 'ch1_config', 'speaker_side']).mean()
        ch_avgs.index = ch_avgs.index.rename('ch_config',level='ch1_config')
    if channel == 3:
        ch_avgs = ch_avgs.loc['extrach']
        ch_avgs = ch_avgs.groupby(['mouse','amplitude', 'ch3_config', 'speaker_side']).mean()
        ch_avgs.index = ch_avgs.index.rename('ch_config',level='ch3_config')
    ABR_stats = []
    for ABRidx in np.arange(0, len(ch_avgs)):
        if ABRidx == len(ch_avgs):
            singleABR = ch_avgs[ABRidx:]
        singleABR = ch_avgs[ABRidx:ABRidx + 1]

        pospk_xs, pospk_ys, negpk_xs, negpk_ys = single_row_extrema(singleABR.values[0], ms_l)
        all_pk_xs = np.concatenate([pospk_xs, negpk_xs])
        all_pk_ys = get_peak_ys(all_pk_xs, np.array(singleABR)[0])

        d = {
            # "info":         singleABR.index,
            "pk_x": all_pk_xs.astype(int),
            "pk_y": all_pk_ys
        }
        pk_info = pandas.DataFrame(d)
        pk_info['speaker_side'] = singleABR.index.get_level_values('speaker_side').values[0]
        pk_info['amplitude'] = singleABR.index.get_level_values('amplitude').values[0]
        pk_info['ch_config'] = singleABR.index.get_level_values('ch_config').values[0]
        pk_info['mouse'] = singleABR.index.get_level_values('mouse').values[0]
        ABR_stats.append(pk_info)
    all_pks_df = pandas.concat(ABR_stats).sort_values(['mouse', 'amplitude', 'ch_config', 'speaker_side', 'pk_x']).reset_index(
        drop=True)
    all_pks_df = all_pks_df.set_index(['mouse', 'amplitude', 'ch_config', 'speaker_side']).sort_index()
    return all_pks_df

def get_singleABR_stddevs(ABR_df):
    # Takes in the full ABR_df of all peaks picked from get_peak_coords_df
    # Returns the peaks after the sound and their std devs away from the mean
    # 0 ms is at ms_l[80]
    baseline_test = ABR_df.loc[ABR_df['pk_x'] <= 80]
    baseline_mean = baseline_test['pk_y'].mean()
    base_std = baseline_test['pk_y'].std()
#    baseline_test.insert(0,'stddevs',(baseline_mean - baseline_test['pk_y']) / base_std)
#    baseline_test['stddevs'] = (baseline_mean - baseline_test['pk_y']) / base_std
    ABR_df.insert(2, 'zscore', ((baseline_mean - ABR_df['pk_y']) / base_std))
    ABR_df.insert(2, 'z_abs', abs(ABR_df['zscore']))
    return ABR_df,baseline_mean,base_std

def get_ABR_data_paths():
    computer = socket.gethostname()
    if computer == 'squid':
        # Rowan's lab desktop
        LV_directory = os.path.expanduser('~/mnt/cuttlefish/abr/LVdata')
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
        Metadata_directory = os.path.expanduser('~/scripts/scripts/rowan/ABR_data')
    elif computer == 'DESKTOP-BIEUSUU':
        # The surgery computer
        LV_directory = os.path.normpath(os.path.expanduser('~/LVdata'))
        Metadata_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/mouse/Documents/GitHub/scripts/rowan/ABR_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser('~/Pickle Temporary Storage'))
    elif computer == 'NSY-0183-PC':
        # Rowan's new laptop
        LV_directory = 'None-- work from pickles'
        Metadata_directory = os.path.normpath(os.path.expanduser
            ('C:/Users/kgargiu/Documents/GitHub/scripts/rowan/ABR_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgargiu/Documents/GitHub/pickles'))
    elif computer=='Theseus':
        LV_directory = 'None-- work from pickles'
        Metadata_directory = os.path.normpath(os.path.expanduser
            ('C:/Users/kgarg/Documents/GitHub/scripts/rowan/ABR_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgarg/Documents/GitHub/pickles'))
    else:
        LV_directory = os.path.expanduser('~/mnt/cuttlefish/abr/LVdata')
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
        Metadata_directory = os.path.expanduser('~/dev/scripts/rowan/ABR_data')
    return LV_directory,Metadata_directory,Pickle_directory
