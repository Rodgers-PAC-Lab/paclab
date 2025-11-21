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

# Three versions of this one - here, abr_gui.py, loading.py
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

