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

    
    ## Load headers
    header_df = pandas.read_table(header_file, sep=',')

    # Extract packet number and unwrap it
    packet_number = np.unwrap(header_df['packet_num'], period=256)

    # Assert no tearing
    assert (np.diff(packet_number) == 1).all()

    
    ## Load raw data
    data = np.fromfile(data_file, dtype=int).reshape(-1, n_channels)

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
    computer = socket.gethostname()
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
    else:
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/Pickle_Temporary_Storage'))
    return GUIdata_directory,Pickle_directory

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

    elif metadata_version=='v5':
        metadata['recording_name'] = ['{:03d}'.format(n) for n in metadata['recording'].values]

    else:
        print("ERROR, METADATA VERSION NOT V4 or V5")
        return
    # Form the full path to the session
    metadata['datafile'] = [
        os.path.join(data_directory, recording_name, 'data.bin')
        for recording_name in metadata['recording_name']]
    return metadata
