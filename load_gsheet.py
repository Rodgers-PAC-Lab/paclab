## Module for loading google sheets

import requests
import pandas

def cedric_ad_sheet():
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
        Columns: [
            'Session Date', 'Mouse Name', 'Video_filename', 'Analog_file', 
            'Recording number', 'Logger', 'Logger_file', 'Behavior_file', 
            'Kilosort_Folder', 'Manually_Sorted', 'Number of neurons',
            'Broken Channels based on tetrode notes', 'Broken on view', 
            'Notes', 'Exclude', 'truncate_last_N_samples', 'Mouse Name  '
            ]
    """
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
        'Visual_channel_table', 'Mice',
        ]    
    
    # Skip these columns .. usually redundant with sheet name
    skip_columns = [
        'Mouse Name'
        ]
    
    # Get data
    request_data = requests.get(url)
    
    # Parse each sheet into a dict
    res_d = {}
    with pandas.ExcelFile(request_data.content) as excel_file:
        # Iterate over sheets
        for sheet_name in excel_file.sheet_names:
            # Skip these
            if sheet_name in skip_sheets:
                continue
            
            # Read this sheet
            sheet = pandas.read_excel(excel_file, sheet_name)
            
            # Store
            res_d[sheet_name] = sheet
    
    # Concatenate along rows
    # This assume the column names are commensurable!
    df = pandas.concat(res_d, names=['mouse_name', 'row'])
    
    # Drop skip_columns
    df = df.drop(skip_columns, axis=1, errors='ignore')
    
    return df
    
