import math
import os.path
import struct
import numpy as np
import scipy
import pickle
import pandas
import socket


def get_metadata(notes_directory,data_directory,datestring):
    ## Get the metadata from a day of recordings
    csv_filename = os.path.join(notes_directory,'20'+datestring+'_ABR', datestring + '_notes.csv')
    metadata = pandas.read_csv(csv_filename)

    # Drop the ones that we don't want to include
    metadata['include'] = metadata['include'].fillna(1).astype(bool)
    metadata = metadata.loc[metadata['include'].values]

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

def extract_onsets_from_audio_data(audio_data, audio_threshold,
                                   abr_start_sample, abr_stop_sample,
                                   winsize=3):
    """Extract the stimulus onset time from the audio data

    The audio data is rectified and smoothed with a window of size `winsize`.
    The times when this smoothed signal crosses `audio_threshold` are
    identified and returned. Only upward crossings are identified.

    audio_data : 1d array

    audio_threshold : float
        Values in `audio_data` exceeding this threshold will be taken as
        a stimulus. This is done on the smoothed data, so for short pulses,
        smoothed data may be much lower htan original data.

    abr_start_sample, abr_stop_sample: int
        `abr_start_sample` should typically be negative.
        Onsets less than `-abr_start_sample` will be discarded.
        Onsets within `abr_stop_sample` of the end of `audio_data` will
        be discarded.
        This allows you to take a window of [abr_start_sample, abr_stop_sample]
        around each onset without running over the end of `audio_data`.

    winsize: int
        How many samples to use in smoothing

    Returns : array
        The onsets of the stimuli, in samples
    """
    # Rectify and smooth
    smoothed_audio_data = pandas.Series(audio_data).abs().rolling(
        winsize, center=True).mean().fillna(0).values

    # Find onsets
    # It should increase by at least `audio_threshold` over no more than
    # `diffsize` frames. Then there's a refractory period of `refrac` samples.
    onsets, durations = extract_onsets_and_durations(
        smoothed_audio_data,
        delta=audio_threshold,
        diffsize=(winsize // 2),
        refrac=50,
        verbose=False,
        maximum_duration=10,
    )

    # ~ # Find threshold crossings
    # ~ # Noise level is around .008
    # ~ # Presently onsets are between 8 and 20 ms apart (1500-3000 samples)
    # ~ onsets = np.where(
    # ~ (smoothed_audio_data[:-1] < audio_threshold) &
    # ~ (smoothed_audio_data[1:] > audio_threshold)
    # ~ )[0]

    # Drop onsets too close to start or end
    onsets = onsets[
        (onsets > -abr_start_sample) &
        (onsets < len(audio_data) - abr_stop_sample)
        ]

    return onsets
def extract_onsets_and_durations(lums, delta=30, diffsize=3, refrac=5,
                                 verbose=False, maximum_duration=100):
    """Identify sudden, sustained increments in the signal `lums`.

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
        print(onsets)

    # drop refractory onsets, offsets
    onsets2 = drop_refrac(onsets, refrac)
    offsets2 = drop_refrac(offsets, refrac)
    if verbose:
        print(onsets2)

    # get durations
    remaining_onsets, durations = extract_duration_of_onsets2(onsets2, offsets2)
    if verbose:
        print(remaining_onsets)

    # apply maximum duration mask
    if maximum_duration is not None:
        max_dur_mask = durations <= maximum_duration
        remaining_onsets = remaining_onsets[max_dur_mask].copy()
        durations = durations[max_dur_mask].copy()

    return remaining_onsets, durations
def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]
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

def parse_by_date(notes_directory, data_directory, datestring, header_size,
    channels, audio_threshold, abr_params):
    """Parses all of the LV binaries from a certain date
    
    Arguments
        audio_threshold : numeric
            stimulus onsets are detected when audio data exceeds this threshold
        
        abr_params : dict, with the following items:
            
            audio_drop_threshold : numeric
                Drop any trials where abs(audio data) exceeds this threshold
            
            neural_drop_threshold : numeric
                Drop any trials where abs(neural data) exceeds this threshold
    """

    #Unpack dicts
    audio_drop_threshold = abr_params["audio_drop_threshold"]
    neural_drop_threshold = abr_params["neural_drop_threshold"]
    abr_start_sample = abr_params["abr_start_sample"]
    abr_stop_sample = abr_params["abr_stop_sample"]
    abr_fig_ymin_uV = abr_params["abr_fig_ymin_uV"]
    abr_fig_ymax_uV = abr_params["abr_fig_ymax_uV"]
    highpass_freq = abr_params["highpass_freq"]
    lowpass_freq = abr_params["lowpass_freq"]
    filter_type = abr_params["filter_type"]
    abr_n_recent_trials = abr_params["abr_n_recent_trials"]
    neural_channel = channels["neural_channel"]
    speaker_channel = channels["speaker_channel"]
    #Get the metadata
    metadata = get_metadata(notes_directory,data_directory,datestring)
    ## Load data from all files
    # Store results here
    triggered_ad_l = []
    triggered_ad_hp_l = []
    triggered_neural_l = []
    keys_l = []

    # Iterate over rows in metadata
    for metadata_idx in metadata.index:
        ## Get the name of the data file
        datafile = metadata.loc[metadata_idx, 'datafile']
        session_name = metadata.loc[metadata_idx, 'session_name']
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
            first_header_info['number_samples'],
        )
        ## Extract the audio onsets
        # Extract audio data
        audio_data = data[:, speaker_channel]

        # de-median it
        audio_data = audio_data - np.median(audio_data)

        # Extract onsets
        onsets = extract_onsets_from_audio_data(
            audio_data, audio_threshold,
            abr_start_sample, abr_stop_sample)

        # Skip if no onsets detected
        if len(onsets) == 0:
            print("warning: {} contains no onsets".format(datafile))
            continue

        # Extract each trigger from the raw audio
        # A 0.1 ms click is only a few samples long, but it rings for ~3 ms at 4 kHz
        # The first ring is 20x smaller than the click, and then it decays
        triggered_ad = np.array([
            audio_data[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])


        ## Extract neural data locked to onsets
        # Extract data
        neural_data = data[:, neural_channel] * 1e6

        # Highpass or bandpass filter
        nyquist_freq = sampling_rate / 2.
        if filter_type == "highpass":
            ahi, bhi = scipy.signal.butter(2, highpass_freq / nyquist_freq,
                                       btype='high')
        else:
            ahi, bhi = scipy.signal.butter(2, [highpass_freq / nyquist_freq,
                lowpass_freq / nyquist_freq], btype='band')
        neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data)

        # Filter the audio data in the same way I GUESS?????
        audio_data_hp = scipy.signal.filtfilt(ahi, bhi, audio_data)
        triggered_ad_hp = np.array([
            audio_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])
        # Extract highpassed neural data around triggers
        triggered_neural = np.array([
            neural_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])

        ## Define time course in ms
        t_plot = np.arange(
            abr_start_sample, abr_stop_sample) / sampling_rate * 1000

        ## Dataframe the results
        triggered_ad_df = pandas.DataFrame(triggered_ad, columns=t_plot)
        triggered_ad_df.index.name = 'trial'
        triggered_ad_df.columns.name = 'timepoint'

        triggered_ad_hp_df = pandas.DataFrame(triggered_ad_hp, columns=t_plot)
        triggered_ad_hp_df.index.name = 'trial'
        triggered_ad_hp_df.columns.name = 'timepoint'

        triggered_neural_df = pandas.DataFrame(triggered_neural, columns=t_plot)
        triggered_neural_df.index.name = 'trial'
        triggered_neural_df.columns.name = 'timepoint'

        
        ## Remove rows with missing packets
        # Identify rows with missing packets
        # This is any row where the audio data exceeds audio_drop_threshold
        # or neural data exceeds neural_drop_threshold
        bad_audio_mask = (
            triggered_ad_df.abs() > audio_drop_threshold).any(axis=1)
        bad_neural_mask = (
            triggered_neural_df.abs() > neural_drop_threshold).any(axis=1)
        bad_mask = bad_audio_mask | bad_neural_mask
        
        # Drop from both neural and audio
        triggered_ad_df = triggered_ad_df.loc[~bad_mask]
        triggered_neural_df = triggered_neural_df.loc[~bad_mask]
        triggered_ad_hp_df = triggered_ad_hp_df.loc[~bad_mask]



        ## Store
        triggered_ad_l.append(triggered_ad_df)
        triggered_ad_hp_l.append(triggered_ad_hp_df)
        triggered_neural_l.append(triggered_neural_df)
        keys_l.append(session_name)
    
    
    ## Concat
    big_ad = pandas.concat(triggered_ad_l, keys=keys_l, names=['session'])
    big_ad_hp = pandas.concat(triggered_ad_hp_l, keys=keys_l, names=['session'])
    big_neural = pandas.concat(triggered_neural_l, keys=keys_l,
                               names=['session'])

    # Add mouse and side to index of big_neural
    to_join = metadata.set_index('session_name')[['mouse','pre_vs_post',
                                                  'speaker_side', 'amplitude']]
    bn_idx = big_neural.index.to_frame().reset_index(drop=True)
    bn_idx = bn_idx.join(to_join, on='session')
    big_neural.index = pandas.MultiIndex.from_frame(bn_idx)
    big_neural = big_neural.reorder_levels(
        ['mouse', 'pre_vs_post','speaker_side', 'amplitude', 'session', 'trial']).sort_index()

    return big_neural,metadata,big_ad, big_ad_hp


def parse_by_date_includeaudio(notes_directory, data_directory, datestring,
        header_size, channels, audio_threshold, abr_params):
    """Parses all of the LV binaries from a certain date

    Arguments
        audio_threshold : numeric
            stimulus onsets are detected when audio data exceeds this threshold

        abr_params : dict, with the following items:

            audio_drop_threshold : numeric
                Drop any trials where abs(audio data) exceeds this threshold

            neural_drop_threshold : numeric
                Drop any trials where abs(neural data) exceeds this threshold
    """

    # Unpack dicts
    audio_drop_threshold = abr_params["audio_drop_threshold"]
    neural_drop_threshold = abr_params["neural_drop_threshold"]
    abr_start_sample = abr_params["abr_start_sample"]
    abr_stop_sample = abr_params["abr_stop_sample"]
    abr_fig_ymin_uV = abr_params["abr_fig_ymin_uV"]
    abr_fig_ymax_uV = abr_params["abr_fig_ymax_uV"]
    highpass_freq = abr_params["highpass_freq"]
    lowpass_freq = abr_params["lowpass_freq"]
    filter_type = abr_params["filter_type"]
    abr_n_recent_trials = abr_params["abr_n_recent_trials"]
    neural_channel = channels["neural_channel"]
    speaker_channel = channels["speaker_channel"]
    # Get the metadata
    metadata = get_metadata(notes_directory, data_directory, datestring)
    ## Load data from all files
    # Store results here
    triggered_ad_l = []
    triggered_ad_hp_l = []
    triggered_neural_l = []
    keys_l = []

    # Iterate over rows in metadata
    for metadata_idx in metadata.index:
        ## Get the name of the data file
        datafile = metadata.loc[metadata_idx, 'datafile']
        session_name = metadata.loc[metadata_idx, 'session_name']
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
            first_header_info['number_samples'],
        )
        ## Extract the audio onsets
        # Extract audio data
        audio_data = data[:, speaker_channel]

        # de-median it
        audio_data = audio_data - np.median(audio_data)

        # Extract onsets
        onsets = extract_onsets_from_audio_data(
            audio_data, audio_threshold,
            abr_start_sample, abr_stop_sample)

        # Skip if no onsets detected
        if len(onsets) == 0:
            print("warning: {} contains no onsets".format(datafile))
            continue

        # Extract each trigger from the raw audio
        # A 0.1 ms click is only a few samples long, but it rings for ~3 ms at 4 kHz
        # The first ring is 20x smaller than the click, and then it decays
        triggered_ad = np.array([
            audio_data[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])

        ## Extract neural data locked to onsets
        # Extract data
        neural_data = data[:, neural_channel] * 1e6

        # Highpass or bandpass filter
        nyquist_freq = sampling_rate / 2.
        if filter_type == "highpass":
            ahi, bhi = scipy.signal.butter(2, highpass_freq / nyquist_freq,
                                           btype='high')
        else:
            ahi, bhi = scipy.signal.butter(2, [highpass_freq / nyquist_freq,
                                               lowpass_freq / nyquist_freq],
                                           btype='band')
        neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data)

        # Filter the audio data in the same way I GUESS?????
        audio_data_hp = scipy.signal.filtfilt(ahi, bhi, audio_data)
        triggered_ad_hp = np.array([
            audio_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])
        # Extract highpassed neural data around triggers
        triggered_neural = np.array([
            neural_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
            for trigger in onsets])

        ## Define time course in ms
        t_plot = np.arange(
            abr_start_sample, abr_stop_sample) / sampling_rate * 1000

        ## Dataframe the results
        triggered_ad_df = pandas.DataFrame(triggered_ad, columns=t_plot)
        triggered_ad_df.index.name = 'trial'
        triggered_ad_df.columns.name = 'timepoint'

        triggered_ad_hp_df = pandas.DataFrame(triggered_ad_hp, columns=t_plot)
        triggered_ad_hp_df.index.name = 'trial'
        triggered_ad_hp_df.columns.name = 'timepoint'

        triggered_neural_df = pandas.DataFrame(triggered_neural, columns=t_plot)
        triggered_neural_df.index.name = 'trial'
        triggered_neural_df.columns.name = 'timepoint'

        ## Remove rows with missing packets
        # Identify rows with missing packets
        # This is any row where the audio data exceeds audio_drop_threshold
        # or neural data exceeds neural_drop_threshold
        bad_audio_mask = (
                triggered_ad_df.abs() > audio_drop_threshold).any(axis=1)
        bad_neural_mask = (
                triggered_neural_df.abs() > neural_drop_threshold).any(axis=1)
        bad_mask = bad_audio_mask | bad_neural_mask

        # Drop from both neural and audio
        triggered_ad_df = triggered_ad_df.loc[~bad_mask]
        triggered_neural_df = triggered_neural_df.loc[~bad_mask]
        triggered_ad_hp_df = triggered_ad_hp_df.loc[~bad_mask]

        ## Store
        triggered_ad_l.append(triggered_ad_df)
        triggered_ad_hp_l.append(triggered_ad_hp_df)
        triggered_neural_l.append(triggered_neural_df)
        keys_l.append(session_name)

    ## Concat
    big_ad = pandas.concat(triggered_ad_l, keys=keys_l, names=['session'])
    big_ad_hp = pandas.concat(triggered_ad_hp_l, keys=keys_l, names=['session'])
    big_neural = pandas.concat(triggered_neural_l, keys=keys_l,
                               names=['session'])

    # Add mouse and side to index of big_neural
    to_join = metadata.set_index('session_name')[['mouse', 'pre_vs_post',
                                                  'speaker_side', 'amplitude']]
    bn_idx = big_neural.index.to_frame().reset_index(drop=True)
    bn_idx = bn_idx.join(to_join, on='session')
    big_neural.index = pandas.MultiIndex.from_frame(bn_idx)
    big_neural = big_neural.reorder_levels(
        ['mouse', 'pre_vs_post', 'speaker_side', 'amplitude', 'session',
         'trial']).sort_index()

    # Separate loud vs quiet trials and insert into all df
    loud_ls = (big_ad > 0.3).any(axis=1).values
    big_ad.insert(0, 'loud', loud_ls)
    big_ad_hp.insert(0, 'loud', loud_ls)
    big_neural.insert(0, 'loud', loud_ls)

   #Prepare dfs so they can be concatted together
    big_ad = big_ad.reset_index()
    big_ad = big_ad.set_index(['session', 'trial', 'loud'])
    bigad_idx = big_ad.index.to_frame().reset_index(drop=True)
    bigad_idx = bigad_idx.join(to_join, on='session')
    big_ad.index = pandas.MultiIndex.from_frame(bigad_idx)
    big_ad = big_ad.reorder_levels(
        ['mouse', 'pre_vs_post', 'speaker_side', 'amplitude', 'session',
         'trial', 'loud']).sort_index()

    big_neural = big_neural.reset_index()
    big_neural = big_neural.set_index(['mouse', 'pre_vs_post', 'speaker_side',
                                       'amplitude', 'session', 'trial', 'loud'])

    big_ad_hp = big_ad_hp.reset_index()
    big_ad_hp = big_ad_hp.set_index(['session', 'trial', 'loud'])
    bigad_hp_idx = big_ad_hp.index.to_frame().reset_index(drop=True)
    bigad_hp_idx = bigad_hp_idx.join(to_join, on='session')
    big_ad_hp.index = pandas.MultiIndex.from_frame(bigad_hp_idx)
    big_ad_hp = big_ad_hp.reorder_levels(
        ['mouse', 'pre_vs_post', 'speaker_side', 'amplitude', 'session',
         'trial', 'loud']).sort_index()

    # Concat audio and neural dfs and sort indexes
    big_concatted = pandas.concat((big_neural, big_ad),
                                keys=['neural', 'audio'])
    big_concatted = big_concatted.sort_index(axis=0, level=['session', 'trial'])
    big_concatted_hp = pandas.concat((big_neural, big_ad_hp),
                                keys=['neural', 'audio'])
    big_concatted_hp = big_concatted_hp.sort_index(axis=0,
                                                level=['session', 'trial'])


    return big_concatted, big_concatted_hp, metadata

def get_ABR_data_paths():
    computer = socket.gethostname()
    if computer == 'squid':
        LV_directory = os.path.expanduser('~/mnt/cuttlefish/abr/LVdata')
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
        Metadata_directory = os.path.expanduser('~/scripts/scripts/rowan/ABR_data')
    elif computer == 'Athena':
        LV_directory = 'None-- work from pickles'
        Metadata_directory = os.path.normpath(os.path.expanduser
            ('C:/Users/kgarg/Documents/GitHub/scripts/rowan/ABR_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgarg/Documents/Career/PAC Lab Stuff/ABR/Figs_Pickles/Concatted Pickles'))
    else:
        LV_directory = os.path.expanduser('~/mnt/cuttlefish/abr/LVdata')
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
        Metadata_directory = os.path.expanduser('~/dev/scripts/rowan/ABR_data')        
    return LV_directory,Metadata_directory,Pickle_directory


