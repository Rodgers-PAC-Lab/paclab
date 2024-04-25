"""For parsing HDF5 files and returning DataFrames"""
import datetime
import os
import numpy as np
import tables
import pandas
import glob
import json
import my
import paclab
import pytz
import scipy.stats
import tqdm

def load_data(
    include_sessions=None, 
    mouse_names=None, 
    protocol_name='PAFT', 
    quiet=False, 
    ):
    """Simple loading function
    
    This is intended to replace boilerplate code in loading behavior data.
    TODO: add sound_data
    
    Steps
    ---
        Gets path to data
        Loads munged sessions
        Calls paclab.parse.parse_sandboxes
        Checks for duplicate sessions and optionally prints warning
        Optionally adds an "n_session" column for each session
    
    
    Parameters
    ---
        include_sessions : list-like or None
            A list of sessions to include
            See deatils in parse_sandboxes
        
        mouse_names : list-like
            A list of mouse names to load
            See details in parse_sandboxes
        
        protocol_name : string or None
            Which protocols to load
            If None, load all
            See details in parse_sandboxes
        
        quiet : bool
            Passed to parse_sandboxes and identify_duplicated_sessions
            If True, prints warning messages to stdout
        
        append_n_session : bool
            If True, appends a column 'n_session' to `session_df` and 
            `perf_metrics`
    
    
    Returns: dict with the following keys
    ---
        perf_metrics : DataFrame from parse_sandboxes
        session_df : DataFrame from parse_sandboxes
        trial_data : DataFrame from parse_sandboxes
        poke_data : DataFrame from parse_sandboxes
        parsing_warnings : string
            From parse_sandboxes
        dup_sessions_warnings : string
            From identify_duplicated_sessions
    """
    ## Get data for parse
    # Get path to terminal data
    path_to_terminal_data = paclab.paths.get_path_to_terminal_data()

    # Munged sessions
    munged_sessions_df = pandas.read_excel(os.path.expanduser(
        '~/mnt/cuttlefish/shared/munged_sessions.xlsx'))
    munged_sessions = munged_sessions_df['Session Name'].values

    # Get parsed_results
    parsed_results = paclab.parse.parse_sandboxes(
        path_to_terminal_data, 
        include_sessions=include_sessions,
        mouse_names=mouse_names,
        munged_sessions=munged_sessions,
        protocol_name=protocol_name,
        quiet=quiet,
        )
    
    # Extract the parsed_results into their own variables
    perf_metrics = parsed_results['perf_metrics'].copy()
    session_df = parsed_results['session_df'].copy()
    trial_data = parsed_results['trial_data'].copy()
    poke_data = parsed_results['poke_data'].copy()
    sound_data = parsed_results['sound_data'].copy()
    parsing_warnings = parsed_results['txt_output']

    # Check for duplicated sessions
    dup_sessions_warnings = identify_duplicated_sessions(
        session_df, quiet=quiet)
    
    # Return data
    return {
        'perf_metrics': perf_metrics,
        'session_df': session_df,
        'trial_data': trial_data,
        'poke_data': poke_data,
        'sound_data': sound_data,
        'dup_sessions_warnings': dup_sessions_warnings,
        'parsing_warnings': parsing_warnings,
        }

def choose_sandboxes_to_enter(
    sandbox_root_dir, 
    include_sessions=None, 
    mouse_names=None, 
    munged_sessions=None):
    """Identify which sandboxes to enter
    
    sandbox_root_dir : str
    
    include_sessions, mouse_names, munged_sessions :
        passed directly from parse_sandboxes
    
    Returns: DataFrame
        columns:
            full_path : full path to sandbox
            sandbox_name : name of sandbox (last bit of full_path)
            sandbox_dt_string, mouse_name : extracted from sandbox_name
            in_munged : True if in munged_sessions
            enter_sandbox : True if should be entered
                This will be True based on include_sessions, mouse_names,
                and munged_sessions
    """
    
    # Get list of available sandbox directories
    sandbox_dir_l = sorted(
        map(os.path.abspath, glob.glob(os.path.join(sandbox_root_dir, '*/'))))
    
    # Form DataFrame
    sandbox_df = pandas.Series(sandbox_dir_l).rename('full_path').to_frame()
    
    # Extract sandbox name
    # This works because of abspath above
    sandbox_df['sandbox_name'] = sandbox_df['full_path'].apply(
        lambda s: os.path.split(s)[1])
    
    # Extract sandbox_dt_string and mouse name
    sandbox_df['sandbox_dt_string'] = sandbox_df['sandbox_name'].str[:26]
    sandbox_df['mouse_name'] = sandbox_df['sandbox_name'].str[27:]
    
    # Decide whether to include via `include_sessions` or `mouse_names`
    # It's not clear what makes sense if both are specified, so in that
    # case just ignore `mouse_names`.
    if include_sessions is not None:
        # In this case, include only the requested sessions
        sandbox_df['enter_sandbox'] = sandbox_df['sandbox_name'].isin(
            include_sessions)
    
    elif mouse_names is not None:
        # In this case, include all sessions from these mice
        sandbox_df['enter_sandbox'] = sandbox_df['mouse_name'].isin(
            mouse_names)
    
    else:
        # While technically they may want to load everything, in practice
        # this is much more likely to be a mistake
        raise ValueError(
            "at least one of `include_sessions` or `mouse_names` "
            "must be specified")

    # Identify munged
    if munged_sessions is not None:
        sandbox_df['in_munged'] = sandbox_df['sandbox_name'].isin(
            munged_sessions)
    else:
        sandbox_df['in_munged'] = np.zeros(len(sandbox_df), dtype=bool)

    # Mask "enter_sessions" by not "in_munged"
    sandbox_df['enter_sandbox'] = (
        sandbox_df['enter_sandbox'] & ~sandbox_df['in_munged'])
    
    return sandbox_df

def check_for_corrupted_trial_data(
    sandbox_name,
    trial_data,
    txt_output,
    quiet,
    ):
    """Check for corrupted trials
    
    sandbox_name : sandbox name
    trial_data : trial data
    txt_output : string
        This is a list of all error messages so far
        It may be appended to
    quiet : if True, print to stdout
    
    Checks for the following:
        * Any trial whose timestamp_trial_start is b''
          If so, it is dropped
        * Extraneous trial numbers - anything outside the range from
          0 to length of trial_data
        * Missing trial numbers - anything not in that range
        * Duplicate trial numbers
    
    The only time trials are dropped is the first case, because
    a busted timestamp will cause other problems downstream
    
    Returns: txt_output, trial_data
        txt_output may have new errors appended
        trial_data may have rows dropped
    """
    # Check for blank start times
    # This seems to co-occur with missing trials, e.g. the first row
    # is corrupted like this, and then the second row is trial 10 or 
    # whatever. In recent memory this has only happened on 2023-02-16
    blank_start_time_mask = trial_data['timestamp_trial_start'] == b''
    if np.any(blank_start_time_mask):
        # Form the text to print
        txt_to_print = (
            'warning: {} has blank start times. dropping\n'.format(
            sandbox_name))
        
        # Do the drop
        trial_data = trial_data[~blank_start_time_mask].copy()
        
        # Print if no quiet
        if not quiet:
            print(txt_to_print.strip())
        
        # Store the text
        txt_output += txt_to_print

    # This is which trials we *should* find
    correct_range = np.array(
        range(trial_data['trial_in_session'].max() + 1))
    
    # These are trials that were found, but should not be there
    # I don't think this is actually possible unless there are negative
    # trial numbers
    extraneous_trials_mask = ~np.isin(
        trial_data['trial_in_session'].values, correct_range)
    
    if np.any(extraneous_trials_mask):
        txt_to_print = ('warning: {} has extraneous trials: {}\n'.format(
            sandbox_name,
            trial_data['trial_in_session'].values[extraneous_trials_mask]))
        if not quiet:
            print(txt_to_print.strip())
        txt_output += txt_to_print

    # These are trials that were not found, but should have been
    trials_not_found_mask = ~np.isin(
        correct_range, trial_data['trial_in_session'].values)
    
    if np.any(trials_not_found_mask):
        txt_to_print = ('warning: {} has missing trials: {}\n'.format(
            sandbox_name,
            correct_range[trials_not_found_mask]))            
        if not quiet:
            print(txt_to_print.strip())
        txt_output += txt_to_print   

    # These are trial numbers that occurred more than once
    duplicate_trials_mask = trial_data['trial_in_session'].duplicated()
    
    if np.any(duplicate_trials_mask):
        txt_to_print = ('warning: {} has duplicate trials: {}\n'.format(
            sandbox_name,
            trial_data['trial_in_session'].values[duplicate_trials_mask],
            ))
        if not quiet:
            print(txt_to_print.strip())
        txt_output += txt_to_print            

    return txt_output, trial_data

def decode_and_coerce_df(
    df,
    columns_to_decode,
    columns_to_timestamp,
    columns_to_bool,
    tz,
    ):
    """Decode and coerce columns
    
    df : DataFrame
    columns_to_decode : list of str
        All of these will be decode('utf-8')
    columns_to_timestamp : list of str
        All of these will be converted to datetime and localized to tz
    columns_to_bool : list of str
        All of these will be converted 'True' to True
    
    Returns: new version of df
    """
    # Copy to avoid view
    df = df.copy()
    
    # Decode
    for decode_col in columns_to_decode:
        df[decode_col] = df[decode_col].str.decode('utf-8')
    
    # Timestamp and localize
    for timestamp_col in columns_to_timestamp:
        df[timestamp_col] = df[timestamp_col].apply(
            datetime.datetime.fromisoformat).dt.tz_localize(tz)
    
    # Bool
    for bool_col in columns_to_bool:
        df[bool_col] = df[bool_col].replace({
            'True': True, 'False': False}).astype(bool)
    
    return df

def handle_sandbox_error_messages(
    skipped_for_broken_hdf5,
    skipped_for_no_trials,
    skipped_for_no_pokes,
    txt_output,
    quiet,
    ):
    """Set up error messages
    
    If there is anything in the first three lists, they will be formatted
    and added to txt_output. Also if not quiet, they will be printed
    
    Returns: txt_output
        May have new messages appended
    """
    if len(skipped_for_broken_hdf5) > 0:
        txt_to_print = (
            "warning: skipped the following sessions with broken HDF5, "
            "these should be added to 'munged_sessions:'\n")
            
        for session_name in skipped_for_broken_hdf5:
            txt_to_print += ('"{}",'.format(session_name)) + '\n'
            
        txt_to_print += '\n'
        
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print            
        
    if len(skipped_for_no_trials) > 0:
        txt_to_print = (
            "warning: skipped the following sessions with zero trials, "
            "these should be added to 'munged_sessions:'\n")
            
        for session_name in skipped_for_no_trials:
            txt_to_print += ('"{}",'.format(session_name)) + '\n'

        txt_to_print += '\n'
        
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print            
    
    if len(skipped_for_no_pokes) > 0:
        txt_to_print = (
            "warning: skipped the following sessions with zero pokes, "
            "these should be added to 'munged_sessions:'\n")
            
        for session_name in skipped_for_no_pokes:
            txt_to_print += ('"{}",'.format(session_name)) + '\n'

        txt_to_print += '\n'
        
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print            
    
    return txt_output

def process_big_session_params(big_session_params, big_trial_df):
    """Add columns and clean big_session_params
    
    * Adds 'date' columns from 'session_name'
    * Adds 'n_trials', 'first_trial', 'last_trial', 'approx_duration',
      'approx_duration_hdf5'
    * Sets 'protocol_name' with assign_protocol_name
    """
    # Specifically: date, approx_duration_hdf5, trial counts and times,
    # and protocol_name
    # The following columns may be null, if they were missing from JSON
    #   protocol_filename, sandbox_creation_time, camera_name
    big_session_params['date'] = [datetime.date.fromisoformat(s[:10]) 
        for s in big_session_params.index.get_level_values('session_name').values]
    
    # Add trial quantifications to big_session_params
    gobj = big_trial_df.groupby(['mouse', 'session_name'])
    big_session_params['n_trials'] = gobj.size()
    big_session_params['first_trial'] = gobj['timestamp_trial_start'].min()
    big_session_params['last_trial'] = gobj['timestamp_trial_start'].max()
    big_session_params['approx_duration'] = (
        big_session_params['last_trial'] - big_session_params['first_trial'])
    
    # Alternate estimate of session duration
    big_session_params['approx_duration_hdf5'] = (
        big_session_params['hdf5_modification_time'] - 
        big_session_params['first_trial'])
    
    # Set protocol_name from protocol_filename
    assign_protocol_name(big_session_params)
    
def assign_protocol_name(big_session_params):
    """Assign a non-null protocol_name to all sessions
    
    Before 2022-10-07, we stored only protocol_name, and afterward
    we stored only protocol_filename. 
    
    For sessions after 2022-10-07, this function calculates what 
    protocol_name was (by dropping the path and extension of 
    protocol_filename), and stores in the column 'protocol_name'.
    
    For sessions before that date, no change is made. 
    
    Returns: big_session_params, with changes made in place.
    """
    # Before 2022-10-07, we only stored protocol_name, and aftewards
    # we only stored protocol_filename
    if 'protocol_filename' in big_session_params.columns:
        # This means we are analyzing at least one session after 2022-10-07
        if 'protocol_name' in big_session_params.columns:
            # This means we are analyzing a mix of data before and after 10-07
            assert ((
                big_session_params['protocol_filename'].isnull().astype(int) + 
                big_session_params['protocol_name'].isnull().astype(int)
                ) == 1).all()
        else:
            # This means we are only analyzing data after 2022-10-07
            # Create this column so we can assign to it below
            big_session_params['protocol_name'] = ''

        # Generate a short protocol name, which is the filename without the
        # full path and without the extension
        short_protocol_name = big_session_params['protocol_filename'].dropna().apply(
            lambda s: os.path.split(s)[1].replace('.json', ''))
        
        # Assign this into protocol_name, which we verified above was either
        # null for the relevant rows, or we just created the column
        big_session_params.loc[
            short_protocol_name.index, 'protocol_name'] = short_protocol_name.values
    
    # Regardless of the above, this should not be null now, unless we are
    # analyzing REALLY old data (?)
    assert not big_session_params['protocol_name'].isnull().any()

def warn_if_no_mouse_found(
    mouse_names,
    all_encountered_mouse_names,
    txt_output,
    quiet,
    ):
    """Warn if no mice were found
    
    mouse_names : list of mouse looked for
    all_encountered_mouse_names : list of mice founod
    txt_output : str to append to
    quiet : bool
    
    Returns: txt_output
        May have warning appended
    """
    if mouse_names is not None:
        missing_mice = []
        for mouse in mouse_names:
            if mouse not in all_encountered_mouse_names:
                missing_mice.append(mouse)
        if len(missing_mice) > 0:
            txt_to_print = "warning: the following mice were not found:\n"
            txt_to_print += "\n".join(missing_mice)
            txt_to_print += '\n'
            txt_to_print += "did you mean one of the following?\n"
            txt_to_print += ", ".join(sorted(all_encountered_mouse_names)) + '\n'
            txt_to_print += '\n'

            if not quiet:
                print(txt_to_print)
            txt_output += txt_to_print                  
    
    return txt_output

def clean_big_trial_df(big_trial_df):
    """Cleans up big_trial_df
    
    * Renames timestamp_trial_start to trial_start
    * Drops group, session, session_uuid
    * Calculates "duration" of each trial, which will always be null
      on the last trial of each session
    
    Returns: big_trial_df (not changed in place)
    """
    # Rename
    big_trial_df = big_trial_df.rename(
        columns={'timestamp_trial_start': 'trial_start'})

    # Drop, these are not relevant
    big_trial_df = big_trial_df.drop(
        ['group', 'session', 'session_uuid'], axis=1)

    # Add duration
    # This shift avoids shifting in data from another session
    # It relies on the index being ['mouse', 'session_name', 'trial'] here
    next_trial_start = big_trial_df.groupby(
        ['mouse', 'session_name'])['trial_start'].shift(-1)
    
    # Duration will be null on the last trial of a session, because there is
    # no next_trial_start
    big_trial_df['duration'] = (
        next_trial_start - big_trial_df['trial_start']).apply(
        lambda ts: ts.total_seconds())
    
    return big_trial_df

def clean_big_poke_df(big_poke_df, big_trial_df, txt_output, quiet):
    """Clean big_poke_df
    
    * Drop pokes from before the first trial, after the last trial,
      or any unaligned with any trial (the last is rare)
    * Join 'trial_start', 'rewarded_port', 'previously_rewarded_port'
      from big_trial_df onto big_poke_df
    * Assert that none of these are null, and previously_rewarded_port is
      null only if it's the first trial
    * Calculate t_wrt_start
    
    Returns: big_poke_df
    """
    # Put trial on the index
    big_poke_df = big_poke_df.set_index(
        'trial', append=True).reorder_levels(
        ['mouse', 'session_name', 'trial', 'poke']).sort_index()        
    
    # Drop all pokes from trial == -1, which I think occurred before
    # the session truly started
    big_poke_df = big_poke_df.drop(-1, level='trial', errors='ignore')
    
    # big_poke_df routinely contains pokes from the trial after the
    # last one in big_trial_df. These are just the pokes from the 
    # never-completed trial (I think?)
    # Drop those pokes
    trial_to_drop = big_trial_df.groupby(
        ['mouse', 'session_name']).apply(
        lambda df: df.index.get_level_values('trial')[-1]).rename(
        'trial') + 1
    
    # Convert to MultiIndex
    trial_to_drop = pandas.MultiIndex.from_frame(trial_to_drop.reset_index())
    
    # Drop pokes from those trials
    big_poke_df = big_poke_df.drop(trial_to_drop, errors='ignore')
    
    # Join 'trial_start' onto big_poke_df
    big_poke_df = big_poke_df.join(big_trial_df[
        ['trial_start', 'rewarded_port', 'previously_rewarded_port']])
    
    # Drop pokes without a matching trial start
    # This hapens very rarely (only on 2023-02-16)
    bad_mask = big_poke_df['trial_start'].isnull()
    if np.any(bad_mask):
        # Group by session
        bad_sessions = big_poke_df[bad_mask].groupby('session_name').size()
        
        # Form the text to print
        txt_to_print = (
            "warning: dropping pokes from {} ".format(len(bad_sessions)) + 
            "sessions that don't have a matching trial: \n{}\n".format(
            str(bad_sessions)))
        
        # Do the drop
        big_poke_df = big_poke_df[~bad_mask].copy()
        
        # Print if no quiet
        if not quiet:
            print(txt_to_print)
        
        # Store the text
        txt_output += txt_to_print

    # After doing this, previously_rewarded_port should ONLY be ''
    # on the first trial
    assert not big_poke_df[
        ['trial_start', 'rewarded_port', 'previously_rewarded_port']
        ].isnull().any().any()
    assert not (
        big_poke_df.drop(0, level='trial')['previously_rewarded_port'] 
        == '').any()

    # Normalize poke time to trial start time
    big_poke_df['t_wrt_start'] = (
        big_poke_df['timestamp'] - big_poke_df['trial_start'])
    big_poke_df['t_wrt_start'] = big_poke_df['t_wrt_start'].apply(
        lambda ts: ts.total_seconds())

    return big_poke_df, txt_output

def reorder_pokes_by_timestamp(big_poke_df):
    """Reorder pokes by timestamp
    
    Original poke index is stored as poke_orig_idx.
    The new 'poke' level on the index is now sorted correctly.
    
    Returns: big_poke_df
    """
    # Copy
    big_poke_df = big_poke_df.copy()
    
    # First pop out the 'poke' index as 'poke_orig'. It has missing indices
    #   because of the dropping above and no longer functions as an index.
    big_poke_df['poke_orig_idx'] = big_poke_df.index.get_level_values('poke')

    # Then argsort poke_orig to get poke_orig_order
    big_poke_df['poke_orig_order'] = big_poke_df.groupby(
        ['mouse', 'session_name'])['poke_orig_idx'].apply(
        lambda ser: ser.droplevel(['mouse', 'session_name']).argsort())
    
    # Then argsort timestamp to get true_poke_order
    # If they arrived out of order, that's probably just networking
    # This happens 0.002 of pokes, on 20% of sessions, especially when 
    # a port is being artefactually activated really quickly
    big_poke_df['true_poke_order'] = big_poke_df.groupby(
        ['mouse', 'session_name'])['timestamp'].apply(
        lambda ser: ser.droplevel(['mouse', 'session_name']).argsort()
        )

    # Replace 'poke' with 'true_poke_order' on index
    big_poke_df = big_poke_df.set_index(
        'true_poke_order', append=True).reset_index(
        'poke', drop=True).sort_index()
    
    # Drop 'poke_orig_order'
    big_poke_df = big_poke_df.drop('poke_orig_order', axis=1)
    
    # Rename index
    big_poke_df.index.names = ['mouse', 'session_name', 'trial', 'poke']

    return big_poke_df

def label_poke_types(big_poke_df, big_trial_df):
    """Label the type of each poke
    
    * Count pokes per trial, save as big_trial_df['n_pokes'], mark
      'unpoked' trials in big_trial_df
    * Set 'poke_type'
    * Set 'choice_poke' (and drop 'first_poke')
    * Set 'consummatory'
    * Error check that all pokes from trials with no choice_poke
      are consummatory
    * Add no_choice_made column to big_trial_df
    * Add no_correct_pokes column to big_trial_df
    
    Returns: big_poke_df, big_trial_df
    """
    ## Copy to avoid weird view
    big_poke_df = big_poke_df.copy()
    big_trial_df = big_trial_df.copy()


    ## Count pokes per trial and check there was at least 1 on every trial
    # Count pokes
    n_pokes = big_poke_df.groupby(
        ['mouse', 'session_name', 'trial']).size().rename('n_pokes')
    
    # Join on big_trial_df
    big_trial_df = big_trial_df.join(n_pokes)
    big_trial_df['n_pokes'] = big_trial_df['n_pokes'].fillna(0).astype(int)
    
    # Very rarely there is a trial with no pokes
    # Mostly these are on 2023-02-16
    # But in two cases it happened on the last trial of the session --
    # Presumably there just wasn't time for a poke
    big_trial_df['unpoked'] = False
    big_trial_df.loc[big_trial_df['n_pokes'] == 0, 'unpoked'] = True
    

    ## Label the type of each poke
    # target port is 'correct'
    # prev_target port is 'prev' (presumably consumptive)
    # all others (currently only one other type) is 'error'
    big_poke_df['poke_type'] = 'error'
    big_poke_df.loc[
        big_poke_df['poked_port'] == big_poke_df['rewarded_port'],
        'poke_type'] = 'correct'
    big_poke_df.loc[
        big_poke_df['poked_port'] == big_poke_df['previously_rewarded_port'],
        'poke_type'] = 'prev'


    ## Identify the choice_poke
    # This is the mouse's "choice"
    first_non_prev_poke = big_poke_df[
        big_poke_df['poke_type'] != 'prev'].reset_index('poke').groupby(
        ['mouse', 'session_name', 'trial']).first()['poke']
    
    # Mark this as the choice_poke
    big_poke_df['choice_poke'] = False
    big_poke_df.loc[
        pandas.MultiIndex.from_frame(first_non_prev_poke.reset_index()),
        'choice_poke'] = True
    
    # Drop 'first_poke', which is calculated on the rpi
    # It turns out this can be wrong in several ways, such as being a 'prev'
    # poke, or for multiple pokes on a trial
    big_poke_df = big_poke_df.drop('first_poke', axis=1)
    
    
    ## Categorize 'prev' pokes into consummatory or not
    # Join the poke index of the choice poke
    # This will be null on trials with no choice poke
    # (i.e., no pokes, or all prev pokes)
    big_poke_df = big_poke_df.join(
        first_non_prev_poke.rename('choice_poke_idx'))    
    
    # Mark all pokes after the choice poke
    # This will be false on trials with no choice poke
    big_poke_df['after_choice_poke'] = False
    after_choice_mask = (
        big_poke_df.index.get_level_values('poke') > 
        big_poke_df['choice_poke_idx'])
    big_poke_df.loc[after_choice_mask, 'after_choice_poke'] = True
    
    # Subcategorize 'prev' pokes into 'prev_consume' and 'prev_return'
    # For backwards compatibility, use another column for this instead of
    # another 'poke type'
    big_poke_df['consummatory'] = False
    prev_consume_mask = (
        (big_poke_df['poke_type'] == 'prev') &
        ~big_poke_df['after_choice_poke'])
    big_poke_df.loc[prev_consume_mask, 'consummatory'] = True

    # Drop 'choice_poke_idx' and 'after_choice_poke'
    big_poke_df = big_poke_df.drop(
        ['choice_poke_idx', 'after_choice_poke'], axis=1)

    # Error check
    # All pokes from trials lacking a choice poke should be consummatory
    # Such trials are extremely rare
    trials_with_choice_poke_mask = big_poke_df.groupby(
        ['mouse', 'session_name', 'trial'])['choice_poke'].any()
    trials_with_no_choice_poke_midx = trials_with_choice_poke_mask.index[
        ~trials_with_choice_poke_mask.values]
    pokes_on_such_trials = my.misc.slice_df_by_some_levels(
        big_poke_df, trials_with_no_choice_poke_midx)
    assert pokes_on_such_trials['consummatory'].all()
    assert (pokes_on_such_trials['poke_type'] == 'prev').all()

    # Label trials with no choice made (which in
    # Include also unpoked trials (which won't show up above because there
    # were no pokes to find)
    big_trial_df['no_choice_made'] = big_trial_df['unpoked'].copy()
    big_trial_df.loc[trials_with_no_choice_poke_midx, 'no_choice_made'] = True


    ## Identify trials lacking a correct poke
    # These are somewhat common, unfortunately, and concentrated into
    # 5-10 sessions, which should likely be dropped
    n_poke_types_by_trial = big_poke_df.groupby(
        ['mouse', 'session_name', 'trial'])['poke_type'].value_counts().unstack(
        'poke_type')
    
    # Reindex by big_trial_df (to include the trials with no pokes)
    n_poke_types_by_trial = n_poke_types_by_trial.reindex(
        big_trial_df.index).fillna(0).astype(int)
    
    # Label trials lacking correct poke
    bad_trials = n_poke_types_by_trial.index[
        n_poke_types_by_trial['correct'] == 0]
    big_trial_df['no_correct_pokes'] = False
    big_trial_df.loc[bad_trials, 'no_correct_pokes'] = True
    
    return big_poke_df, big_trial_df

def label_trial_outcome(big_poke_df, big_trial_df):
    """Label outcome of each trial
    
    * Set big_trial_df['outcome']
    * Set big_poke_df['poke_rank']
    * Set big_trial_df['rcp'] and big_trial_df['first_port_poked']
    
    """
    ## Copy to avoid view
    big_poke_df = big_poke_df.copy()
    big_trial_df = big_trial_df.copy()
    
    
    ## Set outcome
    # Outcome is determined by choice_poke, which cannot be prev by definition
    trial_outcome = big_poke_df.loc[big_poke_df['choice_poke'], 'poke_type']
    
    # Join this on big_trial_df
    # Where it is null, there was no choice
    big_trial_df = big_trial_df.join(
        trial_outcome.rename('outcome').droplevel('poke'))
    big_trial_df['outcome'] = big_trial_df['outcome'].fillna('spoiled')
    

    ## Calculate the rank of each poke, excluding ALL prp
    # Drop this, which was calculated by rpi and often wrong
    big_poke_df = big_poke_df.drop('poke_rank', axis=1)
    
    # Get the latency to each port on each trial, excluding the PRP
    this_poke_df = big_poke_df[big_poke_df['poke_type'] != 'prev']
    latency_by_port = this_poke_df.reset_index().groupby(
        ['mouse', 'session_name', 'trial', 'poked_port'])['t_wrt_start'].min()

    # Unstack the port onto columns
    lbpd_unstacked = latency_by_port.unstack('poked_port')

    # Rank them in order of poking
    # Subtract 1 because it starts with 1
    # The best is 0 (correct trial) and the worst is 6 (because consumption port
    # is ignored). The expectation under random choices is 3
    lbpd_ranked = lbpd_unstacked.rank(
        method='first', axis=1).stack().astype(int) - 1

    # Exception: RCP can be 7 on the first trial of the session, because no PRP
    assert lbpd_ranked.drop(0, level='trial').max() <= 6

    # Join this rank onto big_poke_df
    big_poke_df = big_poke_df.join(lbpd_ranked.rename('poke_rank'), 
        on=['mouse', 'session_name', 'trial', 'poked_port'])

    # Insert poke_rank of -1 wherever poke_type is prev
    assert (
        big_poke_df.loc[big_poke_df['poke_rank'].isnull(), 'poke_type'] 
        == 'prev').all()
    big_poke_df.loc[big_poke_df['poke_type'] == 'prev', 'poke_rank'] = -1
    big_poke_df['poke_rank'] = big_poke_df['poke_rank'].astype(int)

    # Calculate rank of correct poke
    correct_port = big_trial_df['rewarded_port'].dropna().reset_index()
    cp_idx = pandas.MultiIndex.from_frame(correct_port)
    rank_of_correct_port = lbpd_ranked.reindex(
        cp_idx).droplevel('rewarded_port').rename('rcp')

    # Append this to big_trial_df
    big_trial_df = big_trial_df.join(rank_of_correct_port)
    
    # Error check: rcp is null iff no_correct_pokes
    assert big_trial_df.loc[
        big_trial_df['rcp'].isnull()]['no_correct_pokes'].all()
    assert big_trial_df.loc[
        big_trial_df['no_correct_pokes'], 'rcp'].isnull().all()

    # Add the first port poked
    choice_pokes = big_poke_df[big_poke_df['choice_poke']]
    big_trial_df['first_port_poked'] = (
        choice_pokes['poked_port'].droplevel('poke'))

    # Error check: first_port_poked is null iff no_choice_made
    assert big_trial_df.loc[
        big_trial_df['first_port_poked'].isnull()]['no_choice_made'].all()
    assert big_trial_df.loc[
        big_trial_df['no_choice_made'], 'first_port_poked'].isnull().all()    

    return big_poke_df, big_trial_df

def calculate_distance_between_choice_ports(big_poke_df, big_trial_df):
    """Calculate distance between choice, reward, and PRP ports
    
    Adds first_port_poked_dir, previously_rewarded_port_dir, and
    poked_port_dir to big_trial_df
    
    Adds fpp_wrt_rp, fpp_wrt_prp, etc to big_trial_df
    """
    ## Copy to avoid view
    big_trial_df = big_trial_df.copy()
    big_poke_df = big_poke_df.copy()
    

    ## Convert port name to port dir
    big_trial_df = convert_port_name_to_port_dir(big_trial_df)
    big_poke_df = convert_port_name_to_port_dir(big_poke_df)

    
    ## Calculate errdist
    # Define fixing function
    def fix(arr):
        """Fix angular variable to [-180, 180)"""
        return np.mod(arr + 180, 360) - 180
    
    # Apply fix to each
    big_trial_df['fpp_wrt_rp'] = fix(
        big_trial_df['first_port_poked_dir'] - 
        big_trial_df['rewarded_port_dir'])
    big_trial_df['fpp_wrt_prp'] = fix(
        big_trial_df['first_port_poked_dir'] - 
        big_trial_df['previously_rewarded_port_dir'])
    big_trial_df['rp_wrt_prp'] = fix(
        big_trial_df['rewarded_port_dir'] - 
        big_trial_df['previously_rewarded_port_dir'])
    
    big_poke_df['pp_wrt_rp'] = fix(
        big_poke_df['poked_port_dir'] - 
        big_poke_df['rewarded_port_dir'])
    big_poke_df['pp_wrt_prp'] = fix(
        big_poke_df['poked_port_dir'] - 
        big_poke_df['previously_rewarded_port_dir'])

    return big_poke_df, big_trial_df

def calculate_perf_metrics(big_trial_df):
    """Calculate perf_metrics from big_trial_df"""
    # Score sessions by fraction correct
    scored_by_fraction_correct = big_trial_df.groupby(
        ['mouse', 'session_name'])[
        'outcome'].value_counts().unstack('outcome').fillna(0)
    scored_by_fraction_correct['perf'] = (
        scored_by_fraction_correct['correct'].divide(
        scored_by_fraction_correct.sum(axis=1)))

    # Score sessions by n_trials
    scored_by_n_trials = big_trial_df.groupby(
        ['mouse', 'session_name']).size()

    # Score by n_ports
    scored_by_n_ports = big_trial_df.groupby(
        ['mouse', 'session_name'])['rcp'].mean()

    # Extract key performance metrics
    perf_metrics = pandas.concat([
        scored_by_n_ports.rename('rcp'),
        scored_by_fraction_correct['perf'].rename('fc'),
        scored_by_n_trials.rename('n_trials'),
        ], axis=1, verify_integrity=True)

    return perf_metrics

def decode_and_coerce_all_df(trial_data, poke_data, sound_data, tz, old_data=False):
    """Decodes and coerces these dataframes
    
    Doesn't drop any data.
    
    Returns: trial_data, poke_data, sound_data
    """
    # trial_data
    # compatibility with summer 2022 data which lacked 'group'
    if old_data:
        columns_to_decode = ['previously_rewarded_port', 'rewarded_port', 
            'timestamp_reward', 'timestamp_trial_start']
    else:
        columns_to_decode = ['previously_rewarded_port', 'rewarded_port', 
            'timestamp_reward', 'timestamp_trial_start', 'group']
    
    trial_data = decode_and_coerce_df(
        trial_data,
        columns_to_decode=columns_to_decode,
        columns_to_timestamp=['timestamp_reward', 'timestamp_trial_start'],
        columns_to_bool=[],
        tz=tz,
        )
    
    # poke_data
    poke_data = decode_and_coerce_df(
        poke_data,
        columns_to_decode=['poked_port', 'timestamp'],
        columns_to_timestamp=['timestamp'],
        columns_to_bool=['first_poke', 'reward_delivered'],
        tz=tz,
        )        
    
    # sound_data
    if sound_data is not None:
        sound_data = decode_and_coerce_df(
            sound_data,
            columns_to_decode=[
                'pilot', 'side', 'sound_type', 'locking_timestamp'],
            columns_to_timestamp=['locking_timestamp'],
            columns_to_bool=[],
            tz=tz,
            )     
    
    return trial_data, poke_data, sound_data

def warn_about_munged_trials(big_trial_df, txt_output, quiet):
    """Warn about munged trials
    
    Warns about unpoked trials, trials with no choice poke, and trials
    with no correct poke. 
    
    Returns: txt_output
        Potentially with new warnings added
    """
    ## Check for unpoked
    bad_sessions = big_trial_df[
        big_trial_df['unpoked']].groupby('session_name').size()
    
    if len(bad_sessions) > 0:
        txt_to_print = (
            'warning: unpoked trials in {} session(s):\n{}\n'.format(
            len(bad_sessions),
            str(bad_sessions),
            ))

        # Print if no quiet
        if not quiet:
            print(txt_to_print)

        # Store the text
        txt_output += txt_to_print

    
    ## Check for no_choice_madeno_choice_made
    bad_sessions = big_trial_df[
        big_trial_df['no_choice_made']].groupby('session_name').size()
    
    if len(bad_sessions) > 0:
        txt_to_print = (
            'warning: trials without choice in {} session(s):\n{}\n'.format(
            len(bad_sessions),
            str(bad_sessions),
            ))

        # Print if no quiet
        if not quiet:
            print(txt_to_print)

        # Store the text
        txt_output += txt_to_print
    
    
    ## Check for no_correct_pokes
    bad_sessions = big_trial_df[
        big_trial_df['no_correct_pokes']].groupby('session_name').size()
    
    if len(bad_sessions) > 0:
        txt_to_print = (
            'warning: trials w/o correct pokes in {} session(s):\n{}\n'.format(
            len(bad_sessions),
            str(bad_sessions),
            ))

        # Print if no quiet
        if not quiet:
            print(txt_to_print)

        # Store the text
        txt_output += txt_to_print    
    
    return txt_output

def parse_sandboxes(
    path_to_terminal_data, 
    include_sessions=None,
    mouse_names=None, 
    munged_sessions=None,
    protocol_name='PAFT',
    quiet=False,
    load_sound_data=True,
    ):
    """Load the data from the specified mice, clean, and return.
    
    This is a replacement of parse_hdf_files now that data is stored for
    each session in its own "sandbox" instead of in a global mouse HDF5 file.
    
    path_to_terminal_data : string
        The path to Autopilot data. Can be gotten from
        paclab.paths.get_path_to_terminal_data()
    
    include_sessions : list of string, or None
        At least one of `include_sessions` or `mouse_names` must not be None
        If `include_sessions` is not None, then it must be a list, and only
        sessions in that list will be included. Also in this case, `mouse_names`
        is ignored (i.e., this argument takes precedence).
    
    mouse_names : list of string, or None
        At least one of `include_sessions` or `mouse_names` must not be None
        If `include_sesions` is None and `mouse_names` is not None, then 
        only mice in this list will be included.
    
    munged_sessions : list of string
        A list of session names to drop
    
    protocol_name : string or None
        Every session stores the name of the protocol (or task) in 
        task_params.json keyed by task_type. If `protocol_name` is specified
        here, then only sessions for which `task_params[task_type]` matches
        `protocol_name` will be included. 
        If `protocol_name` is None, all sessions are included.
    
    This function:
    * Loads data from sandboxes specified by mouse_names
    * Adds useful columns like t_wrt_start, rewarded_port, etc to pokes
    * Label the type of each poke as "correct", "error", "prev"
    * Label the outcome of the trial as the type of the first poke in that trial
    * Score sessions by fraction correct, rank of correct port
    * Calculated performance metrics over all sessions
    
    TODO: Figure out why some PokeTrain sessions have >100K pokes
    
    Returns: dict, with the following items
        'perf_metrics': DataFrame
            Performance metrics for each session
            Index: MultiIndex with levels mouse, session_name
            Columns: session_name, rcp, fc, n_trials, date, n_session
    
        'session_df' : DataFrame
            Metadata about each session
            Index: MultiIndex with levels mouse, session_name
            Columns: camera_name, pilot, protocol_filename, 
                sandbox_creation_time, hdf5_modification_time, date,
                n_trials, first_trial, last_trial, approx_duration,
                approx_duration_hdf5, protocol_name,
                and all of the task parameters from the protocol file
                and n_session
        
        'trial_data': DataFrame
            Metadata about each trial
            Index: MultiIndex with levels mouse, session_name, trial
            Columns: previously_rewarded_port, rewarded_port, timestamp_reward,
                trial_start, trial_in_session, duration, n_pokes,
                unpoked, no_choice_made,
                no_correct_pokes, outcome, rcp, first_port_poked,
                _dir for PRP, RP, and FPP, fpp_wrt_rp etc,
                and all trial parameters.
        
        'poke_data': DataFrame
            Metadata about each poke
            Index: MultiIndex with levels mouse, session_name, trial, poke
            Columns: poked_port, reward_delivered, timestamp, trial_start, 
                rewarded_port, previously_rewarded_port, t_wrt_start, 
                poke_orig_idx, poke_type, choice_poke, consummatory,
                poke_rank, 
                _dir for poked_port, rewarded_port and choice_poke,
                pp_wrt_rp etc
        
        'sound_data': DataFrame
            Metadata about each sound
            Index: mouse, session_name, sound
            Columns: gap, gap_chunks, locking_timestamp, pilot,
                relative_time, side, sound_type
    """
    ## Initial setup
    # This is used for all timestamps
    tz = pytz.timezone('America/New_York')
    
    # Store text output here for logging
    txt_output = ''
    
    
    ## Decide which directories to enter
    sandbox_root_dir = os.path.join(path_to_terminal_data, 'sandboxes')
    
    # Decide which to enter
    sandbox_df = choose_sandboxes_to_enter(
        sandbox_root_dir, 
        include_sessions=include_sessions, 
        mouse_names=mouse_names, 
        munged_sessions=munged_sessions,
        )

    # Extract all identified mouse names (for a debug message)
    all_encountered_mouse_names = list(sandbox_df['mouse_name'].unique())
    
    # Extract only the sandboxes to enter
    sandboxes_to_enter_df = sandbox_df[sandbox_df['enter_sandbox']]

    
    ## Iterate over sandboxes
    # Create lists to store data
    trial_data_l = []
    poke_data_l = []
    sound_data_l = []
    sandbox_params_l = []
    task_params_l = []
    keys_l = []
    
    # These are things we keep track of
    skipped_for_no_trials = []
    skipped_for_no_pokes = []
    skipped_for_broken_hdf5 = []

    # Iterate over sandboxes
    for sandbox_idx in tqdm.tqdm(sandboxes_to_enter_df.index):
        ## Form sandbox_dir and hdf5_filename
        sandbox_dir = sandboxes_to_enter_df.loc[sandbox_idx, 'full_path']
        sandbox_name = sandboxes_to_enter_df.loc[sandbox_idx, 'sandbox_name']
        mouse_name = sandboxes_to_enter_df.loc[sandbox_idx, 'mouse_name']
        
        # We no longer check that this is the only hdf5 file in the directory
        # because it seems to always be, and takes too long to check
        hdf5_filename = os.path.join(sandbox_dir, sandbox_name + '.hdf5')
        
        
        ## Load json files
        # TODO: deal with what happens if these files don't exist
        with open(os.path.join(sandbox_dir, 'sandbox_params.json')) as fi:
            sandbox_params = json.load(fi)

        with open(os.path.join(sandbox_dir, 'task_params.json')) as fi:
            task_params = json.load(fi)
        
        # Pop this one which is always an empty dict
        task_params.pop('graduation')


        ## Include only the specified task_type
        wrong_task = (
            protocol_name is not None and 
            task_params['task_type'] != protocol_name)

        if wrong_task:
            # Skip
            continue
    
        
        ## Load data from hdf5 file
        # _read_records is the largest single time sink, approx 25% of total 
        try:
            with tables.open_file(hdf5_filename) as fi:
                # Load trial data 
                trial_data = pandas.DataFrame.from_records(
                    fi.root['trial_data'].read())
                
                # We never care about this column
                trial_data = trial_data.drop('trial_num', axis=1)
                
                # Load poke data
                poke_data = pandas.DataFrame.from_records(
                    fi.root['continuous_data']['ChunkData_Pokes'].read())
                
                # Load sound data, or None if doesn't exist (e.g, poketrain)
                # Also skip this if not load_sound_data, to save time
                sound_data = None
                if load_sound_data:
                    try:
                        sound_data = pandas.DataFrame.from_records(
                            fi.root['continuous_data']['ChunkData_Sounds'].read())
                    except IndexError:
                        pass
                
                # Rename column 'sound' which conflicts with index level 'sound'
                if sound_data is not None:
                    sound_data = sound_data.rename(
                        columns={'sound': 'sound_type'})
                

        except (tables.exceptions.HDF5ExtError, FileNotFoundError):
            # This happens on some broken HDF5 files
            # And if it were missing, we would skip it here
            skipped_for_broken_hdf5.append(sandbox_name)
            continue            


        ## Skip sessions with no pokes or no trials
        # This is the last point in this function where we might continue
        # If we pass these, then it will be included in the result
        if len(trial_data) == 0:
            skipped_for_no_trials.append(sandbox_name)
            continue
    
        if len(poke_data) == 0:
            skipped_for_no_pokes.append(sandbox_name)
            continue
    

        ## Add the HDF5 filename mod time to sandbox_params
        # Since we were able to open this hdf5 file, this won't fail
        # Mod time is approximately the end time
        mod_ts = my.misc.get_file_time(hdf5_filename, human=False)
        mod_time = datetime.datetime.fromtimestamp(mod_ts)        
        sandbox_params['hdf5_modification_time'] = mod_time


        ## Further process dataframes
        # Error check for corrupted trial_data
        # This will drop trials with a blank start time, and warn about
        # other types of errors like missing trials
        txt_output, trial_data = check_for_corrupted_trial_data(
            sandbox_name=sandbox_name,
            trial_data=trial_data,
            txt_output=txt_output,
            quiet=quiet,
            )
        
        # Decode and coerce
        trial_data, poke_data, sound_data = decode_and_coerce_all_df(
            trial_data, poke_data, sound_data, tz)


        ## Append
        trial_data_l.append(trial_data)
        poke_data_l.append(poke_data)
        sound_data_l.append(sound_data)
        sandbox_params_l.append(sandbox_params)
        task_params_l.append(task_params)
        keys_l.append((mouse_name, sandbox_name))

    
    ## Warnings
    # Warn about skipped sessions
    txt_output = handle_sandbox_error_messages(
        skipped_for_broken_hdf5,
        skipped_for_no_trials,
        skipped_for_no_pokes,
        txt_output,
        quiet,
        )

    # Warn if no mice found
    txt_output = warn_if_no_mouse_found(
        mouse_names, all_encountered_mouse_names, txt_output, quiet)

    # Error if no sessions found
    if len(sandbox_params_l) == 0:
        raise ValueError(
            "no sandboxes found! either the data cannot be loaded, or "
            "none of your requested mice could be found")
    

    ## Concat big_trial_df, big_poke_df, big_sound_df
    big_trial_df = pandas.concat(
        trial_data_l, keys=keys_l, names=['mouse', 'session_name', 'trial'])
    big_poke_df = pandas.concat(
        poke_data_l, keys=keys_l, names=['mouse', 'session_name', 'poke'])

    # Set this to None if no sound data
    if np.all([val is None for val in sound_data_l]):
        big_sound_df = None
    else:
        big_sound_df = pandas.concat(
            sound_data_l, keys=keys_l, 
            names=['mouse', 'session_name', 'sound'])
    
    
    ## Make big_session_params from task_params and sandbox_params
    big_task_params = pandas.DataFrame.from_records(task_params_l,
        index=pandas.MultiIndex.from_tuples(
        keys_l, names=['mouse', 'session_name']))
    big_sandbox_params = pandas.DataFrame.from_records(sandbox_params_l,
        index=pandas.MultiIndex.from_tuples(
        keys_l, names=['mouse', 'session_name']))

    # Set timezone
    big_sandbox_params['hdf5_modification_time'] = (
        big_sandbox_params['hdf5_modification_time'].dt.tz_localize(tz))

    # Combine big_task_params and big_sandbox_params
    big_session_params = pandas.concat(
        [big_task_params, big_sandbox_params], axis=1, verify_integrity=True)

    
    ## Sort everything that we need going forward
    # From here forward, we only use these four dataframes
    big_trial_df = big_trial_df.sort_index()
    big_poke_df = big_poke_df.sort_index()
    big_session_params = big_session_params.sort_index()
    if big_sound_df is not None:
        big_sound_df = big_sound_df.sort_index()
    
    
    ## Clean up the dataframes by adding columns, etc
    # Adds date, trial counts, and durations to big_session_params
    # Assigns protocol name properly in big_session_params
    process_big_session_params(big_session_params, big_trial_df)
    
    # Removes unnecessary columns and calculates duration
    big_trial_df = clean_big_trial_df(big_trial_df)
    
    # Drops pokes from before the first trial, after last trial, and those
    # that can't be aligned to any trial start
    # Calculates t_wrt_start for each poke
    big_poke_df, txt_output = clean_big_poke_df(
        big_poke_df, big_trial_df, txt_output, quiet)


    ## Label poke type and identify trials with no choice made
    # Sort properly by timestamp (old poke idx is now poke_orig_idx)
    big_poke_df = reorder_pokes_by_timestamp(big_poke_df)
    
    # Label poke types
    # Sets n_pokes and unpoked in big_trial_df
    # Sets poke_type, choice_poke, consummatory in big_poke_df and
    #   drops the flawed 'first_poke'
    # Sets no_choice_made and no_correct_pokes in big_trial_df
    # big_trial_df now includes the follow bool cols:
    #   unpoked: no pokes at all
    #   no_choice_made: no non-PRP pokes
    #   no_correct_pokes: no correct pokes
    # These are all pretty rare. The first two can actually happen (eg on 
    # last trial) but can be artefact. no_correct_pokes is the most common
    # (0.002) and seems to mostly happen aretefactually. 
    # Nothing is dropped at this point
    big_poke_df, big_trial_df = label_poke_types(big_poke_df, big_trial_df)
    
    # Sets 'poke_rank' in big_poke_df and 'rcp', 'outcome', and 
    # 'first_port_poked' in big_trial_df
    # 'rcp' is null on all 'no_correct_pokes'
    # 'first_port_poked' is null and 'outcome' is 'spoiled' on all 'no_choice_made'
    big_poke_df, big_trial_df = label_trial_outcome(big_poke_df, big_trial_df)
    
    # Warn about trials with these errors
    # TODO: drop them?
    txt_output = warn_about_munged_trials(big_trial_df, txt_output, quiet)

    # Convert port names to port dir and calculate err_dist
    big_poke_df, big_trial_df = calculate_distance_between_choice_ports(
        big_poke_df, big_trial_df)


    ## Score sessions into perf_metrics
    perf_metrics = calculate_perf_metrics(big_trial_df)

    # Join date
    perf_metrics = perf_metrics.join(
        big_session_params['date'], on=['mouse', 'session_name'])

    # Join n_session for each mouse
    big_session_params['n_session'] = -1
    for mouse, subdf in big_session_params.groupby('mouse'):
        ranked = subdf['first_trial'].rank(method='first').astype(int) - 1
        big_session_params.loc[ranked.index, 'n_session'] = ranked.values
    assert not (big_session_params['n_session'] == -1).any()

    # Join n_session on perf_metrics
    perf_metrics = perf_metrics.join(big_session_params['n_session'])


    ## Return
    return {
        'session_df': big_session_params,
        'perf_metrics': perf_metrics,
        'trial_data': big_trial_df,
        'poke_data': big_poke_df,
        'sound_data': big_sound_df,
        'txt_output': txt_output,
        }

def identify_duplicated_sessions(session_df, quiet=False):
    """Identify dates with multiple sessions from the same mouse.
    
    Groups `session_df` by 'mouse' and 'date' and checks if this is unique.
    If not, generates human-readable warning messages, which may be printed
    to stdout if `quiet` is False, and in any case are returned as a string.
    
    
    Parameters
    ---
    session_df : DataFrame
        From paclab.parse.parse_sandboxes
    
    quiet : bool
        If True, prints messages to stdout
        Regardless, the messages are returned as a string
    
    
    Returns: string
    ---
        If emtpy, no duplicated sessions found.
        Otherwise, contains human-readable warning messages.
    
    """
    # Store output here
    txt_output = ''
    
    # Identify duplicated sessions
    n_sessions_by_date = session_df.groupby(['mouse', 'date']).size()
    days_with_multiple_sessions = n_sessions_by_date[n_sessions_by_date > 1]

    # Display info about problem dates
    if len(days_with_multiple_sessions) > 0:
        txt_to_print = (
            "warning: some days have multiple sessions from the same mouse!\n"
            "for each case below, add all but 1 session to 'munged_sessions'\n"
            )
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print
        
        # Iterate through each problem case
        for (mouse, date) in days_with_multiple_sessions.index:
            mouse_session_df = session_df.loc[mouse]
            mouse_date_session_df = mouse_session_df.loc[
                mouse_session_df['date'] == date]
            
            txt_to_print = "{} sessions from {} on {}".format(
                len(mouse_date_session_df),
                mouse, date)
            
            txt_to_print += str(
                mouse_date_session_df[
                ['approx_duration', 'n_trials', 'date', 'first_trial']])
            
            txt_to_print += '\n'
            
            if not quiet:
                print(txt_to_print)
            txt_output += txt_to_print
    
    return txt_output

def generate_box_port_dir_df():
    """Returns the direction of each port.
    
    This is copied from autopilot/gui/plots/plot.py, and needs to be
    in sync with that. 
    
    Returns: DataFrame
        index: integers
        columns: box, port, dir
            box: name of parent pi
            port: name of port
            dir: direction of port, with 0 indicating north and 90 east
    """
    box2port_name2port_dir = {
        'rpi_parent01': {
            'rpi09_L': 315,
            'rpi09_R': 0,
            'rpi10_L': 45,
            'rpi10_R': 90,
            'rpi11_L': 135,
            'rpi11_R': 180,
            'rpi12_L': 225,
            'rpi12_R': 270,
        },
        'rpi_parent02': {
            'rpi07_L': 315,
            'rpi07_R': 0,
            'rpi08_L': 45,
            'rpi08_R': 90,
            'rpi05_L': 135,
            'rpi05_R': 180,
            'rpi06_L': 225,
            'rpi06_R': 270,
        },    
        'rpiparent03': {
            'rpi01_L': 225,
            'rpi01_R': 270,
            'rpi02_L': 315,
            'rpi02_R': 0,
            'rpi03_L': 45,
            'rpi03_R': 90,
            'rpi04_L': 135,
            'rpi04_R': 180,
        }, 
        'rpiparent04': {
            'rpi18_L': 90,
            'rpi18_R': 135,
            'rpi19_L': 180,
            'rpi19_R': 225,
            'rpi20_L': 270,
            'rpi20_R': 315,
            'rpi21_L': 0,
            'rpi21_R': 45,
        },
    }    

    # Parse
    ser_d = {}
    for box_name, port_name2port_dir in box2port_name2port_dir.items():
        ser = pandas.Series(port_name2port_dir, name='dir')
        ser.index.name = 'port'
        ser_d[box_name] = ser

    # Concat
    box_port_dir_df = pandas.concat(ser_d, names=['box']).reset_index()
    
    # Convert degrees to cardinal directions
    box_port_dir_df['cardinal'] = box_port_dir_df['dir'].replace({
        0: 'N',
        45: 'NE',
        90: 'E',
        135: 'SE',
        180: 'S',
        225: 'SW',
        270: 'W',
        315: 'NW',
        })
    
    return box_port_dir_df

def convert_port_name_to_port_dir(df):
    """Add columns with port dir based on columns of port names
    
    First uses generate_box_port_dir_df to get the conversion.
    
    Checks for the following column names:
        previously_rewarded_port
        rewarded_port
        first_port_poked
        poked_port
    
    For each of the above, replaces port names with port directions (with
    0 meaning north and 90 east), and stores as a new column of the same name
    with "_dir" appended.
    
    Wherever the conversion fails, np.nan will be used.

    TODO: move this function to paclab repository
    
    Returns: a copy of `df`, with new columns added
    """
    # Get conversion df
    box_port_dir_df = generate_box_port_dir_df()
    
    # Turn this into a replacing dict
    replacing_d = box_port_dir_df.set_index('port')['dir'].to_dict()
    replacing_d[''] = np.nan # use nan for blanks    

    # Copy
    res = df.copy()

    # Columns to check for
    replace_cols = [
        'previously_rewarded_port', 
        'rewarded_port', 
        'first_port_poked',
        'choice_poke',
        'poked_port',
        ]

    # Do each column
    for replace_col in replace_cols:
        if replace_col in res.columns:
            # Replace
            res[replace_col + '_dir'] = res[replace_col].replace(replacing_d)

    return res

def load_sounds_played(h5_filename, session_start_time):
    """Load data about the time every sound was played.
    
    The jack client on every child pi monitors when it is told to play sound.
    It returns this information to the autopilot process on the child pi,
    which forwards that information to the terminal, where it is stored
    as "ChunkData_SoundsPlayed". 
    
    This function loads the "ChunkData_SoundsPlayed" record from the file. 
    Within each row is the datetime that the message was sent, as well as the
    audio "frame number" from jack. We use a linear fit to find a way to 
    convert between frame numbers and datetime.
    
    Then we account for the fact that the sound doesn't play immediately 
    after being entered into the buffer. It plays at the end of the current 
    block of frames, plus N_buffer_blocks later.
    
    After correcting for this delay and converting from frame number to 
    datetime, we have a pretty good estimate of when the sound came out of
    the speaker.
    
    Arguments:
        session_start_time: datetime
            The resulting times will be given in seconds relative to this time.
            Best choice is "trial_start" for the first trial on the parent.
    
    Returns: DataFrame
        index: one for every block of frames where sound was played by any pi
        columns: 
            speaker_time_in_session: time when the sound (should have) come out,
                in seconds since `session_start_time`
            speaker_frame: frame when the sound (should have) come out
            hash: hash of the sound being played in this block
            pilot: pilot that played sound
            message_dt: datetime when the message was sent
            message_time_in_session: time when the message was sent, relative
                to `session_start_time`
            frames_since_cycle_start, last_frame_time, message_frame: 
                indicates the message time in frames        
        
        This DataFrame will be sorted by the column "message_dt"
    """
    ## Load ChunkData_SoundsPlayed
    # Load 
    with tables.open_file(h5_filename) as fi:
        sounds_played_df = pandas.DataFrame.from_records(
            fi.root['continuous_data']['ChunkData_SoundsPlayed'][:])

    # Fix some columns
    # Decode columns that are bytes
    for decode_col in ['equiv_dt', 'pilot']:
        sounds_played_df[decode_col] = (
            sounds_played_df[decode_col].str.decode('utf-8')
            )

    # Coerce timestamp to datetime
    sounds_played_df['message_dt'] = (
        sounds_played_df['equiv_dt'].apply(
        lambda s: datetime.datetime.fromisoformat(s)))
    sounds_played_df = sounds_played_df.drop('equiv_dt', axis=1)

    # tz localize
    tz = pytz.timezone('America/New_York')
    sounds_played_df['message_dt'] = (
        sounds_played_df['message_dt'].dt.tz_localize(tz))

    # Convert equiv_dt to session_time
    # This is necessary because we can't regress to datetime objects
    # Here we take everything relative to session_start_time, which is a little
    # weird because session_start_time came from a different pi (the parent)
    # but should be okay as long as we're consistent
    # These times should be comparable across child pis up to the precision
    # set by chrony (a few ms)
    sounds_played_df['message_time_in_session'] = (
        sounds_played_df['message_dt'] - session_start_time).dt.total_seconds()
    
    
    ## Drop sounds before the session started
    # This shouldn't be possible, but for some reason there are sounds from
    # before the session started. Maybe a leftover network packet?
    drop_mask = sounds_played_df['message_time_in_session'] < 0
    if drop_mask.any():
        print("warning: dropping {} sounds from before session started".format(
            drop_mask.sum()))
        sounds_played_df = sounds_played_df[~drop_mask].copy()
    

    ## Account for buffering delay
    # Calculate message_frame, the frame number at the time the message was sent
    sounds_played_df['message_frame'] = (
        sounds_played_df['last_frame_time'] + 
        sounds_played_df['frames_since_cycle_start'])

    # Calculate the frame when the sound comes out
    # This will be rounded up to the next block, and then plus N_buffer_blocks
    sounds_played_df['speaker_frame'] = (
        (sounds_played_df['message_frame'] // 1024 + 1) * 1024
        + 2 * 1024)

    
    ## Find the best fit between equiv_dt and message_frame
    # Calculate the (almost linear) relationship between jack frame numbers
    # and session time. This must be done separately for each pi since each 
    # has its own frame clock
    #
    # In the course of this fit, we also fix wraparound issues
    pilot2jack_frame2session_time = {}
    new_sounds_played_df_l = []
    for pilot, subdf in sounds_played_df.groupby('pilot'):
        ## Deal with wraparound
        # message_frame can wrap around 2**31 to -2**31
        subdf['message_frame'] = subdf['message_frame'].astype(np.int64)
        #int32_info = np.iinfo(np.int32)
        
        # Detect by this huge offset
        # There can occasionally be small offsets (see below)
        if np.diff(subdf['message_frame']).min() < -.9 * (2**32):
            print("warning: integer wraparound detected in message_frame")
            fix_mask = subdf['message_frame'] < 0
            
            # Fix both message_frame and speaker_frame
            subdf.loc[fix_mask, 'message_frame'] += 2 ** 32
            subdf.loc[fix_mask, 'speaker_frame'] += 2 ** 32
        
        
        ## Error check ordering
        # It is not actually guaranteed that the messages arrive in order
        # 2023-11-29-17-01-27-212505_Fig_BrownLP
        # In one (rare?) case, there were two rows from the same pilot
        # with the same 'speaker_frame' but with two different 'message_frame'
        # Maybe somehow it was trying to play two sounds at once?        
        # And the later one was processed first
        # Warn when this happens
        # As long as its less than a blocksize it's probably (?) okay
        # Certainly indicates the fit can't be perfectly linear, it's not even
        # monotonic
        diff_time = np.diff(subdf['message_frame'])
        n_out_of_order = np.sum(diff_time < 0)
        if n_out_of_order > 0:
            print(
                "warning: {} rows of sounds_played_df ".format(n_out_of_order) +
                "out of order by at worst {} frames".format(diff_time.min())
                )
        
        
        ## Fit from message_frame to session time
        pilot2jack_frame2session_time[pilot] = scipy.stats.linregress(
            subdf['message_frame'].values,
            subdf['message_time_in_session'].values,
        )

        # This should be extremely good because it's fundamentally a link between
        # a time coming from datetime.datetime.now() and a time coming from 
        # jackaudio's frame measurement on the same pi
        # If not, probably an xrun or jack restart occurred
        # 2023-10-30 this got to 1e-10 on 2023-10-25-14-44-43-228389_Fig_BrownLP
        if (1 - pilot2jack_frame2session_time[pilot].rvalue) > 1e-9:
            print("warning: rvalue was {:.3f} on {}".format(
                pilot2jack_frame2session_time[pilot].rvalue,
                pilot))
            print("speaker_time_in_session will be inaccurate")
        
        # It appears the true sampling rate of the Hifiberry is ~192002
        # 1/pilot2jack_frame2session_time[pilot].slope

        
        ## Store the version with times fixed for wraparound
        new_sounds_played_df_l.append(subdf)
    
    # Reconstruct sounds_played_df
    # The ordering is now explicitly sorted by message_dt, whereas before
    # it was sorted by pilot (and I am not sure if it was guaranteed to
    # be sorted by message time within that)
    # Also, message_frame and speaker_frame are now int64 and wraparound-free
    sounds_played_df = pandas.concat(new_sounds_played_df_l).sort_values(
        'message_dt')


    ## Use that fit to estimate when the sound played in the session timebase
    speaker_time_l = []
    for pilot, subdf in sounds_played_df.groupby('pilot'):
        # Convert speaker_time_jack to behavior_time_rpi01
        speaker_time = np.polyval([
            pilot2jack_frame2session_time[pilot].slope,
            pilot2jack_frame2session_time[pilot].intercept,
            ], subdf['speaker_frame'])
        
        # Store these
        speaker_time_l.append(pandas.Series(speaker_time, index=subdf.index))

    # Concat the results and add to sounds_played_df
    concatted = pandas.concat(speaker_time_l)
    sounds_played_df['speaker_time_in_session'] = concatted

    return sounds_played_df

def load_flash_df(h5_filename):
    """Load the flash times from the HDF5 file
    
    On each trial, the parent pi sends a trial start signal to each of the
    four child pis. When they receive it, they flash the LEDs in their
    left and right pokes. We use this flash to synchronize the datastreams,
    because the flash is visible in the video, and also we record the flash
    pulse from one particular pi as an analog input. 
    
    Once the pi receives the trial start signal, it sends a message to the
    terminal containing the timestamp that it receved the signal. These
    timestamps are stored in the HDF5 file. The pis receive the signal
    at slightly different times from each other, because the parent sends
    the signal to each one in turn, and the network delays are unpredictable.
    
    This function loads the records "dt_flash_received" and
    "dt_flash_received_from" from the HDF5 file. These are reshaped into
    trial number on the index and rpi name on the columns. This assumes that
    each pi sent a message on every trial, otherwise this will get messed up.
    As a check for this, we make sure that there's never more than a 200 ms
    difference between the last pi and the first pi on any trial.
    
    Returns : DataFrame
        index: trial number
        columns: rpi name, such as ('rpi01', 'rpi02', 'rpi03', 'rpi04')
        values: the datetime that the message was received by each pi
    """
    # Load flash times
    with tables.open_file(h5_filename) as fi:
        dt_flash_received = pandas.DataFrame.from_records(
            fi.root['continuous_data']['dt_flash_received'][:])
        dt_flash_received_from = pandas.DataFrame.from_records(
            fi.root['continuous_data']['dt_flash_received_from'][:])

    # Form flash times into a single dataframe
    # timestamp and timestamp2 are just the times when the parent received the msg
    flash_df = pandas.concat([
        dt_flash_received.rename(columns={'timestamp': 'timestamp2'}), 
        dt_flash_received_from,
        ], axis=1).rename(columns={'dt_flash_received_from': 'pilot'})
    assert (flash_df['timestamp'] == flash_df['timestamp2']).all()

    # Fix cols
    for decode_col in flash_df.columns:
        flash_df[decode_col] = (
            flash_df[decode_col].str.decode('utf-8')
            )
    for dt_col in ['dt_flash_received', 'timestamp']:
        flash_df[dt_col] = (
            flash_df[dt_col].apply(
            lambda s: datetime.datetime.fromisoformat(s)))

    # drop useless cols
    flash_df = flash_df.drop(['timestamp', 'timestamp2'], axis=1)

    # rearrange, assuming we got all 4 on all trials
    flash_df = flash_df.groupby('pilot').apply(
        lambda df: df['dt_flash_received'].sort_values().reset_index()
        ).drop('index', axis=1)['dt_flash_received'].unstack('pilot')

    # check that this worked, we're not mixing across trials
    assert (flash_df.max(1) - flash_df.min(1)).abs().max().total_seconds() < .2

    # Localize
    tz = pytz.timezone('America/New_York')
    for colname in flash_df.columns:
        flash_df[colname ] = flash_df[colname].dt.tz_localize(tz)

    # these are real trial numbers, unless we missed the first one before recording started
    flash_df.index.name = 'trial'
    
    return flash_df

def process_session_sound_data(session_sound_data, session_trial_data):
    """Given the parsed session_sound_data, calculate when the sounds played
    
    Returns: DataFrame
        session_sound_data with 'absolute_time' column added
    """
    # Make a copy
    session_sound_data = session_sound_data.copy()
    session_trial_data = session_trial_data.copy()
    
    # This assumes that every trial occurs as exactly one locking_timestamp
    # which is probably true
    session_sound_data['trial'] = session_sound_data['locking_timestamp'].rank(
        method='dense').astype(int) - 1

    # The last trial probably didn't get completed (although it could have)
    # So there will be sounds but no trial info
    assert session_sound_data['trial'].max() == len(session_trial_data)

    # Check how close the known trial time is to the locking timestamp
    # Typically locking_timestamp seems to be 350-600ms after trial_start
    session_sound_data['trial_start'] = session_sound_data['trial'].map(session_trial_data['trial_start'])
    session_sound_data['latency'] = session_sound_data['trial_start'] - session_sound_data['locking_timestamp']
    session_trial_data['latency'] = -session_sound_data.groupby('trial')['latency'].mean()

    # Some pis might be slightly faster than others, but only by ~30 ms
    latency_by_port = session_trial_data.groupby('rewarded_port')['latency'].mean().sort_values()

    # Drop this for sanity
    session_sound_data = session_sound_data.drop('latency', axis=1)

    # Reconstruct the sound time
    # 'relative_time' is what was drawn from the distr, but 'gap_chunks' is what
    # was actually used, and is rounded/quantized.
    # When drawing from the distr, the duration of the sound was ignored,
    # so everything is slightly off
    # Also, the gap is AFTER the sound, not before. Ie the first entry in the
    # sound_cycle is a sound, not a gap.
    ssd2_l = []
    for trial, sub_session_sound_data in session_sound_data.groupby('trial'):
        ssd2 = sub_session_sound_data.copy()
        
        # Calculate the time in chunks of each sound, accounting for the fact
        # that the sound itself is 2 chunks long
        # and also accounting for the fact that the first sound plays at time zero
        ssd2['gaptime'] = (ssd2['gap_chunks'] + 2).cumsum().shift().fillna(0).astype(int)
        ssd2['gaptime_s'] = ssd2['gaptime'] * 1024 / 192000

        # Save
        ssd2_l.append(ssd2)

    # Reconstitute
    new_session_sound_data = pandas.concat(ssd2_l)
        
    # Calculate the absolute time of each sound, using the locking timestamp
    # and the newly calculated and corrected gaptime
    # TODO: account for the fact that the sounds repeat after the first cycle
    session_sound_data['absolute_time'] = (
        session_sound_data['locking_timestamp'] + 
        new_session_sound_data['gaptime_s'].apply(lambda x: datetime.timedelta(seconds=x)))
    
    return session_sound_data
