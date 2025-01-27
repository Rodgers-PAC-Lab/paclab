"""Functions to load data from the GUI"""
import os
import json
import numpy as np
import pandas

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