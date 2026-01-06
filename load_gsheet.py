## Module for loading google sheets

import requests
import pandas
import numpy as np
import io
import os

def load(doc_id, sheet_name=None, normalize_case=True):
    """Load a google sheet at the specified url
    
    doc_id : str
        This is a long alphanumeric string, after "spreadsheets/d/" and
        before "/edit", containing no slashes but sometimes hyphens.
    
    sheet_name : str or None
        The name of the sheet (i.e., tab) to load
        If None, the first sheet is loaded
    
    normalize_case : bool
        If True, the case of the column headers is normalized by removing
        spaces and lower-casing
    """

    # Form URL
    # The sheet must be visible to anyone with the link
    # Export as xlsx because the CSV format doesn't support multiple sheets
    url = (
        'https://docs.google.com/spreadsheets/d/' + # google prefix
        doc_id + 
        '/export?format=xlsx' # export command
        )

    # Get data
    request_data = requests.get(url)
    
    # Error check
    if not request_data.ok:
        raise ValueError(
            f'could not load gsheet {doc_id}; '
            f'status {request_data.status_code}; is it publicly shared?')

    # Parse each sheet into a dict
    res_d = {}
    with pandas.ExcelFile(io.BytesIO(request_data.content)) as excel_file:
        # Default to first sheet
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]

        # Read this sheet
        sheet = pandas.read_excel(excel_file, sheet_name)

        # Fix the column names
        if normalize_case:
            sheet.columns = [
                normalize_case_of_string(col) for col in sheet.columns]
        
        # Label row numbers starting with 2 to match google sheet
        sheet.index = sheet.index.values + 2

    return sheet    

def normalize_case_of_string(s):
    """Lower-case string and replace spaces with underscores"""
    return s.lower().replace(' ', '_')

def normalize_list_of_channels(ser):
    """Convert channel numbers data into a tuple of ints or None
    
    ser : pandas.Series to convert
    
    For each value in ser:
        If value is null:
            return None
        If value is string-like
            Separate by commas, convert to a tuple of integers
        Otherwise
            Assume it's a single number and return a tuple comprising just
            that number
    
    Returns: list of new values
    """
    new_values_l = []
    for key, value in ser.items():
        if pandas.isnull(value):
            tup = None
        
        elif hasattr(value, '__len__'):
            # It's probably string-like
            try:
                tup = tuple(sorted(map(int, value.split(','))))
            except ValueError:
                print('warning: cannot convert ' +
                    'the value `{}` to a tuple of integers '.format(value) +
                    'in {} entry {}'.format(ser.name, key)
                    )
                tup = None
        
        else:
            # It might be a single value
            try:
                tup = tuple([int(value)])
            except ValueError:
                print('warning: cannot convert ' +
                    'the value `{}` to a tuple of integers '.format(value) +
                    'in {} entry {}'.format(ser.name, key)
                    )
                tup = None         

        new_values_l.append(tup)
    
    return new_values_l

def abigail_sheet(drop_long_columns=True):
    """Load data from '2025-02-15 Abigail neural recordings metadata' into dict
    
    This is currently a copy of cedric_ad_sheet. Different data cleaning
    may be required. TODO: Figure out which parts of this code can be shared 
    between the two.
    """
    # Where to look for analog path
    analog_path = os.path.expanduser('~/mnt/cuttlefish/whitematter/d_drive')
    
    # Get URL
    # The sheet must be visible to anyone with the link
    # The CSV format doesn't support multiple sheets
    url = (
        'https://docs.google.com/spreadsheets/d/' # google prefix
        '1EDIHYOcQYYXEwoCEpfGCMsPImGwTRgyhQxEr2nClMRs/' # doc ID
        'export?format=xlsx' # export command
        )

    # Skip these sheets
    skip_sheets = [
        'Directories', 'Behavior_mice', 'Corrupted_files', 
        'Visual_channel_table', 'mouse_list', 'Mice',
        ]    
    
    # Skip these columns (in normalized case)
    skip_columns = [
        'mouse_name', # usually redundant with sheet name
        'truncate_last_n_samples', # not using anymore?
        ]
    
    # Optionally skip these
    if drop_long_columns:
        skip_columns += [
            'notes', 
            'broken_on_view', 
            'broken_channels_based_on_tetrode_notes',
            ]
    
    # Get data
    request_data = requests.get(url)

    # Parse each sheet into a dict
    res_d = {}
    with pandas.ExcelFile(io.BytesIO(request_data.content)) as excel_file:
        # Iterate over sheets
        for sheet_name in excel_file.sheet_names:
            # Skip these
            if sheet_name in skip_sheets:
                continue
            
            # Read this sheet
            sheet = pandas.read_excel(excel_file, sheet_name)

            # Fix the column names
            sheet.columns = [
                normalize_case_of_string(col) for col in sheet.columns]
            
            # Label row numbers starting with 2 to match google sheet
            sheet.index = sheet.index.values + 2
            
            # Store
            res_d[sheet_name] = sheet
    
    # Concatenate along rows
    # This assume the column names are commensurable!
    df = pandas.concat(res_d, names=['mouse_name', 'row'])
    
    # Drop skip_columns
    df = df.drop(skip_columns, axis=1, errors='ignore')

    # Set the date column to be a datetime.date instead of a timestamp
    df['session_date'] = df['session_date'].dt.date
    
    # Fix the logger name
    df['logger'] = df['logger'].replace({
        'A62': '62BA62',
        'logger_62BB7C': '62BB7C',
        'logger_62BA62': '62BA62',
        })
    
    # Fix exclude by making it bool with default False
    df['exclude'] = df['exclude'].fillna(0)
    bad_exclude_vals = ~df['exclude'].isin([0, 1])
    if bad_exclude_vals.any():
        print(
            "The following values in 'exclude' need to be replaced with "
            "True or False: ")
        print(df['exclude'].loc[bad_exclude_vals])
        print()
        
        # Assume those bad values do need to be excluded
        df.loc[bad_exclude_vals, 'exclude'] = 1
    df['exclude'] = df['exclude'].astype(bool)
    
    # Fix 'broken_channels_based_on_tetrode_notes'
    # Make it a tuple of integers, or None
    if 'broken_channels_based_on_tetrode_notes' in df.columns:
        df['broken_channels_based_on_tetrode_notes'] = (
            normalize_list_of_channels(
            df['broken_channels_based_on_tetrode_notes']))
    
    # Fix 'broken_on_view'
    # Make it a tuple of integers, or None
    if 'broken_on_view' in df.columns:
        df['broken_on_view'] = normalize_list_of_channels(
            df['broken_on_view'])


    ## Fix manually sorted by making it bool with default False
    # First fill null with 0
    df['manually_sorted'] = df['manually_sorted'].fillna(0)
    
    # Now replace the things I know are in there
    replacing_ser = pandas.Series(
        {'yes': 1, 'Yes': 1, 'No': 0, 'no': 0, 0: 0, 1: 1, True: 1, False: 0})
    bad_vals_mask = ~df['manually_sorted'].isin(replacing_ser.index)
    if bad_vals_mask.any():
        print(
            "The following values in 'manually_sorted' need to be replaced with "
            "True or False: ")
        print(df['manually_sorted'].loc[bad_vals_mask])
        print()
        
        # Assume those bad values were not manually sorted
        df.loc[bad_vals_mask, 'manually_sorted'] = 0
    
    # Now map
    df['manually_sorted'] = df['manually_sorted'].map(replacing_ser).astype(bool)


    ## Fill recording_number with 1, the default value
    if 'recording_number' not in df.columns:
        df['recording_number'] = 1
    df['recording_number'] = df['recording_number'].fillna(1).astype(int)


    ## Figure out analog_packed_filename
    # Add analog_packed_filename
    df['analog_packed_filename'] = ''
    for sta_index in df.index:
        # Slice data from table
        analog_session_name = df.loc[sta_index, 'analog_file']
        if pandas.isnull(analog_session_name):
            continue

        # Recording number within analog data
        recording_number = df.loc[
            sta_index, 'recording_number']
        recording_number_string = 'recording{}'.format(int(recording_number))
        
        # Full path to analog data
        analog_packed_filename = os.path.join(
            analog_path, 
            analog_session_name,
            'Record Node 107/experiment1',
            recording_number_string,
            'continuous/eCube_Server-105.0',
            'continuous.dat',
            )

        # Error check it exists
        assert os.path.exists(analog_packed_filename)
        
        # Store
        df.loc[
            sta_index, 'analog_packed_filename'] = analog_packed_filename

    
    return df
    

def cedric_ad_sheet(drop_long_columns=True):
    """Load data from 'AD Mice Synchronization Table' into dict
    
    This can take some time to run, a few seconds. Sometimes up to 10 s, and
    may even time out. Don't spam the server with multiple requests per 
    second or they may block us.
    
    The sheet ID inside this function will allow anyone to load the data,
    and should be kept private, not put into public-facing version control.
    
    The sheets that are known not to correspond to mouse names are dropped.
    Then the remaining sheets are concatenated together, which means we
    are assuming they have the same column names. This means that
    if other sheets are added to the google doc that aren't mice and have
    other kinds of columns, it will probably break this function.
    
    Returns: DataFrame
        Index: MultiIndex with levels ('mouse_name', 'row')
        Columns: 
            'session_date' : datetime.date
                The date of the session
            'video_filename' : string or null
                Video filename (without full path)
            'analog_file' : string or null
                Directory name of analog data (without full path)
            'recording_number' : float or null
                If not null, the recording number
            'logger' : string or null
                The logger name
                One of: '62BA62', '62BB7C'
                TODO: why is this sometimes null?
            'logger_file': string or null
                If not null, the neural filename (without full path)
            'behavior_file': string or null
                If not null, the behavior filename (without full path)
            'kilosort_folder': string or null
                If not null, the kilosort directory (without full path)
            'manually_sorted' : bool
                True if the original spreadsheet said Yes or yes
                False otherwise, including nulls
            'number_of_neurons': float or null
            'broken_channels_based_on_tetrode_notes': tuple or None
                If a tuple, it will be a list of integers of broken channels
                for this implant
            'broken_on_view': tuple or None
                If a tuple, it will be a list of integer of channels that 
                appeared to broken for this session
            'notes' : string or none
            'exclude' : bool
                True if it was True in the spreadsheet or had weird text in it
                False if it was False in the spreadsheet or blank
            'truncate_last_N_samples' : float or None
                How many samples to truncate from the end before sorting
                Are we still doing this?
            'drop_first_N_analog_triggers' : float or None
                How many analog triggers to drop from the beginning before
                syncing
    """
    # Where to look for analog path
    analog_path = os.path.expanduser('~/mnt/cuttlefish/whitematter/d_drive')
    
    # Get URL
    # The sheet must be visible to anyone with the link
    # The CSV format doesn't support multiple sheets
    url = (
        'https://docs.google.com/spreadsheets/d/' # google prefix
        '1Dvdc8TBz62hi2Qt4_kMgWEwuAeyf81BhZ9TwzzYJrBg/' # doc ID
        'export?format=xlsx' # export command
        )

    # Skip these sheets
    skip_sheets = [
        'Directories', 'Behavior_mice', 'Corrupted_files', 
        'Visual_channel_table', 'mouse_list', 'Mice',
        ]    
    
    # Skip these columns (in normalized case)
    skip_columns = [
        'mouse_name', # usually redundant with sheet name
        'truncate_last_n_samples', # not using anymore?
        ]
    
    # Optionally skip these
    if drop_long_columns:
        skip_columns += [
            'notes', 
            'broken_on_view', 
            'broken_channels_based_on_tetrode_notes',
            ]
    
    # Get data
    request_data = requests.get(url)

    # Parse each sheet into a dict
    res_d = {}
    with pandas.ExcelFile(io.BytesIO(request_data.content)) as excel_file:
        # Iterate over sheets
        for sheet_name in excel_file.sheet_names:
            # Skip these
            if sheet_name in skip_sheets:
                continue
            
            # Read this sheet
            sheet = pandas.read_excel(excel_file, sheet_name)

            # Fix the column names
            sheet.columns = [
                normalize_case_of_string(col) for col in sheet.columns]
            
            # Label row numbers starting with 2 to match google sheet
            sheet.index = sheet.index.values + 2
            
            # Store
            res_d[sheet_name] = sheet
    
    # Concatenate along rows
    # This assume the column names are commensurable!
    df = pandas.concat(res_d, names=['mouse_name', 'row'])
    
    # Drop skip_columns
    df = df.drop(skip_columns, axis=1, errors='ignore')

    # Set the date column to be a datetime.date instead of a timestamp
    df['session_date'] = df['session_date'].dt.date
    
    # Fix the logger name
    df['logger'] = df['logger'].replace({
        'A62': '62BA62',
        'logger_62BB7C': '62BB7C',
        'logger_62BA62': '62BA62',
        })
    
    # Fix exclude by making it bool with default False
    df['exclude'] = df['exclude'].fillna(0)
    bad_exclude_vals = ~df['exclude'].isin([0, 1])
    if bad_exclude_vals.any():
        print(
            "The following values in 'exclude' need to be replaced with "
            "True or False: ")
        print(df['exclude'].loc[bad_exclude_vals])
        print()
        
        # Assume those bad values do need to be excluded
        df.loc[bad_exclude_vals, 'exclude'] = 1
    df['exclude'] = df['exclude'].astype(bool)
    
    # Fix 'broken_channels_based_on_tetrode_notes'
    # Make it a tuple of integers, or None
    if 'broken_channels_based_on_tetrode_notes' in df.columns:
        df['broken_channels_based_on_tetrode_notes'] = (
            normalize_list_of_channels(
            df['broken_channels_based_on_tetrode_notes']))
    
    # Fix 'broken_on_view'
    # Make it a tuple of integers, or None
    if 'broken_on_view' in df.columns:
        df['broken_on_view'] = normalize_list_of_channels(
            df['broken_on_view'])


    ## Fix manually sorted by making it bool with default False
    # First fill null with 0
    df['manually_sorted'] = df['manually_sorted'].fillna(0)
    
    # Now replace the things I know are in there
    replacing_ser = pandas.Series(
        {'yes': 1, 'Yes': 1, 'No': 0, 'no': 0, 0: 0, 1: 1, True: 1, False: 0})
    bad_vals_mask = ~df['manually_sorted'].isin(replacing_ser.index)
    if bad_vals_mask.any():
        print(
            "The following values in 'manually_sorted' need to be replaced with "
            "True or False: ")
        print(df['manually_sorted'].loc[bad_vals_mask])
        print()
        
        # Assume those bad values were not manually sorted
        df.loc[bad_vals_mask, 'manually_sorted'] = 0
    
    # Now map
    df['manually_sorted'] = df['manually_sorted'].map(replacing_ser).astype(bool)


    ## Fill recording_number with 1, the default value
    df['recording_number'] = df['recording_number'].fillna(1).astype(int)


    ## Figure out analog_packed_filename
    # Add analog_packed_filename
    df['analog_packed_filename'] = ''
    for sta_index in df.index:
        # Slice data from table
        analog_session_name = df.loc[sta_index, 'analog_file']
        if pandas.isnull(analog_session_name):
            continue

        # Recording number within analog data
        recording_number = df.loc[
            sta_index, 'recording_number']
        recording_number_string = 'recording{}'.format(int(recording_number))
        
        # Full path to analog data
        analog_packed_filename = os.path.join(
            analog_path, 
            analog_session_name,
            'Record Node 107/experiment1',
            recording_number_string,
            'continuous/eCube_Server-105.0',
            'continuous.dat',
            )

        # Error check it exists
        assert os.path.exists(analog_packed_filename)
        
        # Store
        df.loc[
            sta_index, 'analog_packed_filename'] = analog_packed_filename

    
    return df
    
def munged_sessions():
    """Return the list of munged sessions"""
    sheet = load('1gOlX4hvBkH_MmcGBmANqQg8sLBrOivs_S_ZB7BUfqD8')
    return sheet
