"""Functions to load data from the GUI"""
import os
import json
import numpy as np
import pandas
import socket

def load_recording(recording_directory):
    """Load data from a recording
    
    recording_directory : str
        Path to a directory containing config.json, data.bin, packet_headers.csv
    
    Returns: dict
        config : dict of configuration
        header_df : DataFrame of headers
        data : array of data, in V
    """
    
    ## Form the different file names
    config_file = os.path.join(recording_directory, 'config.json')
    header_file = os.path.join(recording_directory, 'packet_headers.csv')
    data_file = os.path.join(recording_directory, 'data.bin')
    
    ## Open the file
    # Load config
    with open(config_file) as fi:
        config = json.load(fi)

    # Parse params we need from config
    gains = np.array(config['gains'])
    n_channels = len(gains)
    full_range_mV = config['pos_fullscale'] - config['neg_fullscale']
    full_range_V = full_range_mV / 1000

    # This wasn't always stored. When it wasn't stored, the dtype was
    # (mistakenly) int64
    if 'output_dtype' in config:
        assert config['output_dtype'] == 'int32'
        output_dtype = np.int32
    else:
        # This was a mistake in early verions
        output_dtype = int

    
    ## Load headers
    header_df = pandas.read_table(header_file, sep=',')

    # Extract packet number and unwrap it
    header_df['packet_num_unwrapped'] = np.unwrap(
        header_df['packet_num'], period=256)

    # Warn if tearing
    if np.any(np.diff(header_df['packet_num_unwrapped']) != 1):
        print('warning: data is torn')

    
    ## Load raw data
    data = np.fromfile(data_file, dtype=output_dtype).reshape(-1, n_channels)

    # Rescale - the full range is used by the bit_depth
    data = data * full_range_V / 2 ** 24

    # Account for gain that was applied by ADS1299
    data = data / np.array(config['gains'])

    return {
        'config': config,
        'header_df': header_df,
        'data': data,
    }

def get_ABR_data_paths():
    """Set directories for ABR analysis separately for each computer
    
    This script gets the hostname and returns directories to use for
    ABR data and data storage.
    
    Returns: GUIdata_directory, Pickle_directory
        GUIdata_directory : path to location of raw ABR data
        Pickle_directory : path to location where intermediate data files
            and figures can be stored
    """
    # Get the hostname
    computer = socket.gethostname()
    
    # Appropriately set the directories based on hostname
    if computer == 'squid':
        # Rowan's lab desktop
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
    
    elif computer == 'mantaray':
        # The surgery computer- windows partition
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/Pickle_Temporary_Storage'))
    
    elif computer == 'NSY-0183-PC':
        # Rowan's new laptop
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgargiu/Documents/GitHub/pickles'))
        Pickle_directory = GUIdata_directory
    
    elif computer=='Theseus':
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgarg/Documents/GitHub/pickles'))
        Pickle_directory =  GUIdata_directory

    elif computer=='kelp':
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.expanduser(
        '~/mnt/cuttlefish/cedric/data/ABR/pickles')
    
    elif computer == 'cephalopod':
        # Chris' computer
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/chris/data/20250720_abr_data'))
            
    else:
        # defaults
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/Pickle_Temporary_Storage'))
    
    return GUIdata_directory, Pickle_directory

def get_metadata(data_directory, datestring, metadata_version):
    """Get the metadata from a day of recordings

    Looks for the metadata csv file
    Reads this csv file
    Forms a recording_name column
    Forms a datafile column

    Parameters:
        data_directory: where the metadata and the subfolders for recordings live
        datestring: yymmdd datestring
        metadata_version: which version of metadata csv is used (different columns and naming schemes)
    Returns: Metadata as dataframe, with datafile (file path) column added
    """

    # Form the filename to the csv file
    if metadata_version == "v4":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v4.csv')
    elif metadata_version == "v5":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v5.csv')
    elif metadata_version == "v6":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v6.csv')
    else:
        print("ERROR, Metadata csv not found for " + datestring)
        return

    # Read the CSV file
    metadata = pandas.read_csv(csv_filename)
    metadata['include'] = metadata['include'].fillna(1).astype(bool)

    # Deserialize the amplitudes with json
    metadata['amplitude'] = metadata['amplitude'].apply(json.loads)

    # Form ch0_config and make recording_name from session number
    if metadata_version=='v4':
        metadata['recording_name'] = ['{:03d}'.format(n) for n in metadata['session'].values]
        metadata['ch0_config'] = (metadata['positive_electrode'].str.cat(metadata['negative_electrode']))
        metadata['ch2_config'] = (metadata['positive_ch3'].str.cat(metadata['negative_ch3']))
        metadata['ch4_config'] = 'NN'
        metadata = metadata.rename(columns={"session":"recording"})
        metadata = metadata.drop(['pre_vs_post','positive_electrode','negative_electrode','positive_ch3','negative_ch3'],axis=1)
    elif metadata_version=='v5' or metadata_version=='v6':
        metadata['recording_name'] = ['{:03d}'.format(n) for n in metadata['recording'].values]
        if metadata_version=='v5':
            metadata['ch4_config'] = 'NN'
    else:
        print("ERROR, METADATA VERSION NOT V4, V5, OR V6")
        return
    
    # Form the full path to the session
    metadata['datafile'] = [
        os.path.join(data_directory, recording_name)
        for recording_name in metadata['recording_name']]
    
    return metadata

def drop_torn_packets(data,header_df, debug_plot=False,speaker_channel=7):
    # Go through and find where the tearing is
    torn = np.diff(header_df['packet_num_unwrapped']) != 1
    # This labels data as torn ON THE PACKET WHERE IT MESSED UP, instead of the next correct packet
    header_df['torn'] = np.array([*torn, False])
    # To calculate the correct sample numbers we need to know the number of packets recieved by the GUI,
    #  rather than the ACTUAL packet number sent from the Teensy
    header_df['pkt_sequence'] = np.arange(len(header_df))
    header_df['sample_number_start'] = header_df['pkt_sequence'] * 500
    torn_headers = header_df.loc[header_df['torn'] == True]

    # Optional plot of the speaker channel with torn packets highlighted in green
    if debug_plot:
        import matplotlib.pyplot as plt
        plt.plot(data[:, speaker_channel])
        bad_samples = []
        for pkt in torn_headers.index:
            start = torn_headers.loc[pkt]['sample_number_start']
            bad_samples.append([start, start + 500])
            plt.axvspan(start, start + 500, facecolor='green', alpha=0.5)
        plt.show()

    packets_l = []
    for pkt_n in range(len(header_df)):
        pkt = header_df.iloc[pkt_n]
        if pkt['torn'] == False:
            packets_l.append(
                data[(pkt_n * 500): (pkt_n + 1) * 500]
            )
    new_data = np.concatenate(packets_l)

    # Make sure the number of samples dropped matches the number of packets torn
    assert len(data) - len(new_data) == len(torn_headers) * 500

    return new_data, header_df