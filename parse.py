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

def load_data(mouse_names, protocol_name='PAFT'):
    """Simple loading function
    
    This is intended to replace boilerplate code in loading behavior data.
    
    Steps:
        Gets path to data
        Loads munged sessions
        Calls paclab.parse.parse_sandboxes
        Checks for duplicate sessions and prints warning
        Adds an "n_session" column for each session
        Returns the data unpacked from paclab.parse.parse_sandboxes
    
    Arguments (see paclab.parse.parse_sandboxes for details)
        mouse_names : list-like
            A list of mouse names to load
        
        protocol_name : string
            Which protocols to load
    
    Example call:
        perf_metrics, session_df, trial_data, poke_data = load_data(
            mouse_names=mouse_names, protocol_name='PAFT')
    
    Returns:
        perf_metrics, session_df, trial_data, poke_data
    """
    ## Get data for parse
    # Get path to terminal data
    path_to_terminal_data = paclab.paths.get_path_to_terminal_data()

    # Munged sessions
    munged_sessions_df = pandas.read_excel(os.path.expanduser(
        '~/mnt/cuttlefish/shared/munged_sessions.xlsx'))
    munged_sessions = munged_sessions_df['Session Name'].values
    rename_sessions_l = [
        ]

    # Get parsed_results
    parsed_results = paclab.parse.parse_sandboxes(
        path_to_terminal_data, 
        mouse_names=mouse_names,
        rename_sessions_l=rename_sessions_l,
        munged_sessions=munged_sessions,
        protocol_name=protocol_name,
        )

    # Extract the parsed_results into their own variables
    perf_metrics = parsed_results['perf_metrics'].copy()
    session_df = parsed_results['session_df'].copy()
    trial_data = parsed_results['trial_data'].copy()
    poke_data = parsed_results['poke_data'].copy()

    # Identify duplicated sessions
    n_sessions_by_date = session_df.groupby(['mouse', 'date']).size()
    days_with_multiple_sessions = n_sessions_by_date[n_sessions_by_date > 1]

    # Display info about problem dates
    if len(days_with_multiple_sessions) > 0:
        print(
            "warning: some days have multiple sessions from the same mouse!")
        print(
            "for each case below, add all but one session to 'munged_sessions'")
        print()
        
        # Iterate through each problem case
        for (mouse, date) in days_with_multiple_sessions.index:
            mouse_session_df = session_df.loc[mouse]
            mouse_date_session_df = mouse_session_df.loc[
                mouse_session_df['date'] == date]
            print("{} sessions from {} on {}".format(
                len(mouse_date_session_df),
                mouse, date))
            print(mouse_date_session_df[
                ['approx_duration', 'n_trials', 'date', 'first_trial']])
            print()

    # Add n_session for each mouse
    session_df['n_session'] = -1
    for mouse, subdf in session_df.groupby('mouse'):
        ranked = subdf['first_trial'].rank(method='first').astype(int) - 1
        session_df.loc[ranked.index, 'n_session'] = ranked.values
    assert not (session_df['n_session'] == -1).any()

    # Join n_session on perf_metrics
    perf_metrics = perf_metrics.join(session_df['n_session'])

    # Return data
    return perf_metrics, session_df, trial_data, poke_data

def parse_sandboxes(
    path_to_terminal_data, 
    mouse_names=None, 
    munged_sessions=None,
    rename_sessions_l=None,
    rename_mouse_d=None,
    protocol_name='PAFT',
    quiet=False,
    ):
    """Load the data from the specified mice, clean, and return.
    
    This is a replacement of parse_hdf_files now that data is stored
    for each session in its own "sandbox" instead of in a global mouse HDF5
    file.
    
    path_to_terminal_data : string
        The path to Autopilot data. Can be gotten from
        paclab.paths.get_path_to_terminal_data()
    
    mouse_names : list of string, or None
        A list of mouse names to load
        If None, all mice are loaded.
    
    munged_sessions : list of string
        A list of session names to drop
    
    rename_sessions_l : list, or None
        If None, no renaming is done.
        Otherwise, each item in the list is a tuple of length 3
        Each tuple consists of (
            the session_name the file was saved as,
            the session_name it should have been saved as,
            the name of the actual mouse that was tested)
        Example:
            ('20220309115351-M2_PAFT-Box2', 
            '20220309115351-F2_PAFT-Box2', 
            'F2_PAFT',)        
    
    rename_mouse_d : dict or None
        If None, no renaming is done
        Otherwise, for each (key, value) pair, rename the mouse named
        "key" to "value".     

    protocol_name : string or None
        Every session stores the name of the protocol (or task) in 
        task_params.json keyed by task_type. If `protocol_name` is specified
        here, then only sessions for which `task_params[task_type]` matches
        `protocol_name` will be included. 
        If `protocol_name` is None, all sessions are included.
    
    # TODO: Figure out why some PokeTrain sessions have >100K pokes

    This function:
    * Loads data from sandboxes specified by mouse_names
    * Adds useful columns like t_wrt_start, rewarded_port, etc to pokes
    * Label the type of each poke as "correct", "error", "prev"
    * Label the outcome of the trial as the type of the first poke in that trial
    * Score sessions by fraction correct, rank of correct port
    * Calculated performance metrics over all sessions
    
    Returns: dict, with the following items
        'perf_metrics': DataFrame
            Performance metrics for each session
            Index: MultiIndex with levels mouse, date
            Columns: session_name, rcp, fc, n_trials
    
        'session_df' : DataFrame
            Metadata about each session
            Index: MultiIndex with levels mouse, session_name
            Columns: box, date, 
                n_trials, approx_duration, weight,
                task_type, pilot, protocol_name, task_class_name, camera_name,
                sandbox_creation_time, and all of the task parameters from
                the protocol file
            TODO: 
                add "box"
                split task_parameters into a different variable
        
        'trial_data': DataFrame
            Metadata about each trial
            Index: MultiIndex with levels mouse, session_name, trial
            Columns: previously_rewarded_port, rewarded_port, timestamp_reward,
                trial_start, duration, n_pokes, correct, error, prev, 
                outcome, rcp, and all trial parameters
            TODO:
                split trial_parameters into a different variable
        
        'poke_data': DataFrame
            Metadata about each poke
            Index: MultiIndex with levels mouse, session_name, trial, poke
            Columns: poked_port, timestamp, trial_start, t_wrt_start, 
                rewarded_port, previously_rewarded_port, poke_type,
                poke_rank, reward_delivered
            
            TODO:
                add "trial" as a level on the index
    """
    ## Store text output here for logging
    txt_output = ''
    
    
    ## Get list of available sandboxes
    sandbox_root_dir = os.path.join(path_to_terminal_data, 'sandboxes')
    sandbox_dir_l = sorted(
        map(os.path.abspath, glob.glob(os.path.join(sandbox_root_dir, '*/'))))

    
    ## Iterate over sandboxes
    # Create lists to store data
    trial_data_l = []
    poke_data_l = []
    sound_data_l = []
    sandbox_params_l = []
    task_params_l = []
    keys_l = []
    skipped_for_no_hdf5 = []
    skipped_for_no_trials = []
    skipped_for_no_pokes = []
    skipped_for_broken_hdf5 = []
    all_encountered_mouse_names = []

    # Iterate over sandboxes
    for sandbox_dir in sandbox_dir_l:
        ## Parse sandbox names
        # Get sandbox name
        sandbox_name = os.path.split(sandbox_dir)[1]
        
        # Get datetime and mouse_name from sandbox_name
        sandbox_dt_string = sandbox_name[:26]
        mouse_name = sandbox_name[27:]
        
        # Keep track of all mice encountered
        if mouse_name not in all_encountered_mouse_names:
            all_encountered_mouse_names.append(mouse_name)
        
        # Continue if mouse_name not needed
        if mouse_names is not None and mouse_name not in mouse_names:
            continue
        
        # Continue if munged
        if munged_sessions is not None and sandbox_name in munged_sessions:
            continue
        
        # Get HDF5 filename
        hdf5_filename_l = glob.glob(os.path.join(sandbox_dir, '*.hdf5'))
        
        # Skip if no hdf5 files found (and later warn)
        if len(hdf5_filename_l) != 1:
            skipped_for_no_hdf5.append(sandbox_name)
            continue
        
        # Extract unique hdf5 filename
        assert len(hdf5_filename_l) == 1
        hdf5_filename = hdf5_filename_l[0]
        
        
        ## Load json files
        with open(os.path.join(sandbox_dir, 'sandbox_params.json')) as fi:
            sandbox_params = json.load(fi)

        with open(os.path.join(sandbox_dir, 'task_params.json')) as fi:
            task_params = json.load(fi)
        
        # Pop this one which is always an empty dict
        task_params.pop('graduation')
        
        
        ## Add the HDF5 filename mod time to sandbox_params
        # Mod time
        # This is approximately the end time
        mod_ts = my.misc.get_file_time(hdf5_filename, human=False)
        mod_time = datetime.datetime.fromtimestamp(mod_ts)        
        sandbox_params['hdf5_modification_time'] = mod_time
        
        
        ## Include only the specified task_type
        if (
                protocol_name is not None and 
                task_params['task_type'] != protocol_name):
            continue
        
        
        ## Load data from hdf5 file
        try:
            with tables.open_file(hdf5_filename) as fi:
                # Load trial data 
                trial_data = pandas.DataFrame.from_records(
                    fi.root['trial_data'].read())
                
                # Load poke data
                poke_data = pandas.DataFrame.from_records(
                    fi.root['continuous_data']['ChunkData_Pokes'].read())
                
                # Load sound data, or None if doesn't exist (e.g, poketrain)
                try:
                    sound_data = pandas.DataFrame.from_records(
                        fi.root['continuous_data']['ChunkData_Sounds'].read())
                except IndexError:
                    sound_data = None
        except tables.exceptions.HDF5ExtError:
            # This happens on some broken HDF5 files
            skipped_for_broken_hdf5.append(sandbox_name)
            continue            


        ## Sometimes trials have a blank timestamp_trial_start
        # I think this only happens when it crashes right away
        # Drop those trials, hopefully nothing else breaks
        # Check if this is still happening
        trial_data = trial_data[
            trial_data['timestamp_trial_start'] != b''].copy()
        

        ## Drop this one that we never care about
        trial_data = trial_data.drop('trial_num', axis=1)


        ## Skip sessions with no pokes or no trials
        if len(trial_data) == 0:
            skipped_for_no_trials.append(sandbox_name)
            continue
        if len(poke_data) == 0:
            skipped_for_no_pokes.append(sandbox_name)
            continue
        
        
        ## Check for duplicated, missing or extraneous trials
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
                print(txt_to_print)
            txt_output += txt_to_print

        # These are trials that were not found, but should have been
        trials_not_found_mask = ~np.isin(
            correct_range, trial_data['trial_in_session'].values)
        
        if np.any(trials_not_found_mask):
            txt_to_print = ('warning: {} has missing trials: {}\n'.format(
                sandbox_name,
                correct_range[trials_not_found_mask]))            
            if not quiet:
                print(txt_to_print)
            txt_output += txt_to_print            

        # These are trial numbers that occurred more than once
        duplicate_trials_mask = trial_data['trial_in_session'].duplicated()
        
        if np.any(duplicate_trials_mask):
            txt_to_print = ('warning: {} has duplicate trials: {}\n'.format(
                sandbox_name,
                trial_data['trial_in_session'].values[duplicate_trials_mask],
                ))
            if not quiet:
                print(txt_to_print)
            txt_output += txt_to_print            
        
        
        ## Coerce dtypes for trial_data
        # Decode columns that are bytes
        for decode_col in ['previously_rewarded_port', 'rewarded_port', 
                'timestamp_reward', 'timestamp_trial_start', 'group']:
            trial_data[decode_col] = (
                trial_data[decode_col].str.decode('utf-8')
                )

        # Coerce timestamp to datetime
        for timestamp_column in ['timestamp_reward', 'timestamp_trial_start']:
            trial_data[timestamp_column] = (
                trial_data[timestamp_column].apply(
                lambda s: datetime.datetime.fromisoformat(s)))

        # Coerce the columns that are boolean
        bool_cols = []
        for bool_col in bool_cols:
            trial_data[bool_col] = trial_data[bool_col].replace(
                {'True': True, 'False': False}).astype(bool)

        
        ## Coerce dtypes for poke_data
        # Decode columns that are bytes
        for decode_col in ['poked_port', 'timestamp']:
            poke_data[decode_col] = poke_data[
                decode_col].str.decode('utf-8')

        # Coerce timestamp to datetime
        for timestamp_column in ['timestamp']:
            poke_data[timestamp_column] = (
                poke_data[timestamp_column].apply(
                lambda s: datetime.datetime.fromisoformat(s)))

        # Coerce the columns that are boolean
        bool_cols = ['first_poke', 'reward_delivered']
        for bool_col in bool_cols:
            try:
                poke_data[bool_col] = poke_data[bool_col].replace(
                    {1: True, 0: False}).astype(bool)
            except KeyError:
                print("warning: missing bool_col {}".format(bool_col))
        

        ## Coerce dtypes for sound_data
        if sound_data is not None:
            # Rename the column 'sound' which conflicts with index level 'sound'
            sound_data = sound_data.rename(
                columns={'sound': 'sound_type'})        
            
            # Decode columns that are bytes
            for decode_col in ['pilot', 'side', 'sound_type', 'locking_timestamp']:
                sound_data[decode_col] = sound_data[
                    decode_col].str.decode('utf-8')

            # Coerce timestamp to datetime
            for timestamp_column in ['locking_timestamp']:
                sound_data[timestamp_column] = (
                    sound_data[timestamp_column].apply(
                    lambda s: datetime.datetime.fromisoformat(s)))


        ## Append
        trial_data_l.append(trial_data)
        poke_data_l.append(poke_data)
        sound_data_l.append(sound_data)
        sandbox_params_l.append(sandbox_params)
        task_params_l.append(task_params)
        keys_l.append((mouse_name, sandbox_name))


    ## Warn about skipped sessions
    if len(skipped_for_no_hdf5) > 0:
        txt_to_print = (
            "warning: skipped the following sessions with no HDF5 file, "
            "these should be added to 'munged_sessions:'\n")

        for session_name in skipped_for_no_hdf5:
            txt_to_print += ('"{}",'.format(session_name)) + '\n'

        txt_to_print += '\n'
        
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print            

    
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


    ## Warn if no mice found
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
    

    ## Error if no sessions found
    if len(sandbox_params_l) == 0:
        raise ValueError(
            "no sandboxes found! either the data cannot be loaded, or "
            "none of your requested mice could be found")
    

    ## Concat
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
    
    big_task_params = pandas.DataFrame.from_records(task_params_l,
        index=pandas.MultiIndex.from_tuples(keys_l, names=['mouse', 'session_name']))
    big_sandbox_params = pandas.DataFrame.from_records(sandbox_params_l,
        index=pandas.MultiIndex.from_tuples(keys_l, names=['mouse', 'session_name']))
    big_session_params = pandas.concat(
        [big_task_params, big_sandbox_params], axis=1, verify_integrity=True)


    ## Parse big_session_params
    # add 'date' to big_session_params
    big_session_params['date'] = [datetime.date.fromisoformat(s[:10]) 
        for s in big_session_params.index.get_level_values('session_name').values]
    
    # Add trial quantifications to big_session_params
    big_session_params['n_trials'] = big_trial_df.groupby(
        ['mouse', 'session_name']).size()
    big_session_params['first_trial'] = big_trial_df.groupby(
        ['mouse', 'session_name'])[
        'timestamp_trial_start'].min()
    big_session_params['last_trial'] = big_trial_df.groupby(
        ['mouse', 'session_name'])[
        'timestamp_trial_start'].max()
    big_session_params['approx_duration'] = (
        big_session_params['last_trial'] - big_session_params['first_trial'])
    
    # Alternate estimate of session duration
    big_session_params['approx_duration_hdf5'] = (
        big_session_params['hdf5_modification_time'] - 
        big_session_params['first_trial'])
    
    

    ## Parse trial data further
    # Rename
    big_trial_df = big_trial_df.rename(
        columns={'timestamp_trial_start': 'trial_start'})

    # Drop, not sure what this is supposed to be
    big_trial_df = big_trial_df.drop('group', axis=1)

    # Drop, these are not relevant
    big_trial_df = big_trial_df.drop(['session', 'session_uuid'], axis=1)

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


    ## Add columns to big_trial_df, and calculate t_wrt_start for pokes
    # Join 'trial_start' onto big_poke_df
    big_poke_df = big_poke_df.join(
        big_trial_df['trial_start'], on=['mouse', 'session_name', 'trial'])

    # Drop pokes without a matching trial start
    # TODO: Look into why this is happening -- is it only before first trial 
    # and on last trial or what?
    big_poke_df = big_poke_df[~big_poke_df['trial_start'].isnull()].copy()

    # Normalize poke time to trial start time
    big_poke_df['t_wrt_start'] = (
        big_poke_df['timestamp'] - big_poke_df['trial_start'])
    big_poke_df['t_wrt_start'] = big_poke_df['t_wrt_start'].apply(
        lambda ts: ts.total_seconds())


    ## Join rewarded_port and previously_rewarded_port on pokes
    big_poke_df = big_poke_df.join(
        big_trial_df[['rewarded_port', 'previously_rewarded_port']],
        on=['mouse', 'session_name', 'trial'],
        rsuffix='_correct')

    # Actually don't do this because then it looks like no pokes on this trial
    #~ # Drop big_poke_df rows where previously_rewarded_port is ''
    #~ # TODO: Check that this is only the first trial in a session
    #~ # TODO: first trial should be 0 not 1
    #~ big_poke_df = big_poke_df[big_poke_df['previously_rewarded_port'] != ''].copy()

    # Add trial to the index
    big_poke_df = big_poke_df.set_index('trial', append=True).reorder_levels(
        ['mouse', 'session_name', 'trial', 'poke']).sort_index()


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


    ## Count pokes per trial and check there was at least 1 on every trial
    n_pokes = big_poke_df.groupby(
        ['mouse', 'session_name', 'trial']).size().rename('n_pokes')
    big_trial_df = big_trial_df.join(n_pokes)
    #~ assert not big_trial_df['n_pokes'].isnull().any()

    # TODO figure out why this is sometimes null
    big_trial_df = big_trial_df[~big_trial_df['n_pokes'].isnull()].copy()
    big_trial_df['n_pokes'] = big_trial_df['n_pokes'].astype(int)

    # It should never be zero, it would have been null if so
    assert (big_trial_df['n_pokes'] > 0).all()


    ## Debug there is always a correct poke
    n_poke_types_by_trial = big_poke_df.groupby(
        ['mouse', 'session_name', 'trial'])['poke_type'].value_counts().unstack(
        'poke_type').fillna(0).astype(int)
    assert len(big_trial_df) == len(n_poke_types_by_trial)

    # TODO: this is not working, figure out why
    #assert (n_poke_types_by_trial['correct'] > 0).all()


    ## Check whether first_poke was correctly identified
    n_first_pokes_per_trial = big_poke_df.groupby(
        ['session_name', 'trial'])['first_poke'].sum()
    n_first_pokes_per_trial = n_first_pokes_per_trial[
        n_first_pokes_per_trial != 1]
    if len(n_first_pokes_per_trial) > 0:
        txt_to_print = (
            'warning: non-unique first poke in the following trials:\n' + 
            str(n_first_pokes_per_trial) + '\n')
        if not quiet:
            print(txt_to_print)
        txt_output += txt_to_print            
    
    # Fix this
    # I think there are at least 3 ways that first_poke can be wrong
    # First it sometimes is not set for any poke (rarely)
    # Second it sometimes is set for multiple pokes (rarely)
    # Third it might be set for the previously rewarded port 
    first_poke_idx = big_poke_df[big_poke_df['poke_type'] != 'prev'].groupby(
        ['session_name', 'trial'])['timestamp'].idxmin().values
    first_poke_midx = pandas.MultiIndex.from_tuples(
        first_poke_idx, names=big_poke_df.index.names)
    big_poke_df['first_poke2'] = big_poke_df['first_poke'].copy()
    big_poke_df['first_poke2'] = False
    big_poke_df.loc[first_poke_midx, 'first_poke2'] = True
    
    # Replace first_poke with first_poke2
    # In at least one case, this creates a trial that previously had a 
    # first_poke but now no longer does, because it consists entirely of
    # pokes on the previously_rewarded_port, one of which was erroneously
    # labeled first poke initially
    # 2023-02-16-14-52-42-349196_Moon_2 trial 54
    big_poke_df['first_poke'] = big_poke_df['first_poke2']
    big_poke_df = big_poke_df.drop('first_poke2', axis=1)


    ## Label the outcome of each trial, based on the type of the first poke
    # time of first poke of each type
    first_poke = big_poke_df.reset_index().groupby(
        ['mouse', 'session_name', 'trial', 'poke_type']
        )['t_wrt_start'].min().unstack('poke_type')

    # Join
    big_trial_df = big_trial_df.join(first_poke)

    # Score by first poke (excluding prev)
    trial_outcome = big_trial_df[['correct', 'error']].idxmin(1)
    big_trial_df['outcome'] = trial_outcome


    ## Label trials by how many ports poked before correct
    # Get the latency to each port on each trial
    latency_by_port = big_poke_df.reset_index().groupby(
        ['mouse', 'session_name', 'trial', 'poked_port'])['t_wrt_start'].min()

    # Drop the consumption port (previous reward)
    consumption_port = big_trial_df[
        'previously_rewarded_port'].dropna().reset_index().rename(
        columns={'previously_rewarded_port': 'poked_port'})
    cp_idx = pandas.MultiIndex.from_frame(consumption_port)
    latency_by_port_dropped = latency_by_port.drop(cp_idx, errors='ignore')

    # Unstack the port onto columns
    lbpd_unstacked = latency_by_port_dropped.unstack('poked_port')

    # Rank them in order of poking
    # Subtract 1 because it starts with 1
    # The best is 0 (correct trial) and the worst is 6 (because consumption port
    # is ignored). The expectation under random choices is 3 (right??)
    lbpd_ranked = lbpd_unstacked.rank(
        method='first', axis=1).stack().astype(int) - 1

    # TODO: figure out why rcp is sometimes 7 here

    # Find the rank of the correct port
    correct_port = big_trial_df['rewarded_port'].dropna().reset_index()
    cp_idx = pandas.MultiIndex.from_frame(correct_port)
    rank_of_correct_port = lbpd_ranked.reindex(
        cp_idx).droplevel('rewarded_port').rename('rcp')

    # Append this to big_big_trial_df
    big_trial_df = big_trial_df.join(rank_of_correct_port)

    # Error check
    # TODO: figure out why this is not working
    #~ assert not big_trial_df['rcp'].isnull().any()


    ## Score sessions by fraction correct
    scored_by_fraction_correct = big_trial_df.groupby(
        ['mouse', 'session_name'])[
        'outcome'].value_counts().unstack('outcome').fillna(0)
    scored_by_fraction_correct['perf'] = (
        scored_by_fraction_correct['correct'].divide(
        scored_by_fraction_correct.sum(axis=1)))


    ## Score sessions by n_trials
    scored_by_n_trials = big_trial_df.groupby(
        ['mouse', 'session_name']).size()


    ## Score by n_ports
    scored_by_n_ports = big_trial_df.groupby(
        ['mouse', 'session_name'])['rcp'].mean()


    ## Extract key performance metrics
    # This slices out sound-only trials
    perf_metrics = pandas.concat([
        scored_by_n_ports.rename('rcp'),
        scored_by_fraction_correct['perf'].rename('fc'),
        scored_by_n_trials.rename('n_trials'),
        ], axis=1, verify_integrity=True)

    # Join date
    perf_metrics = perf_metrics.join(
        big_session_params['date'], on=['mouse', 'session_name'])

    #~ # Keep the session with the most trials by date
    #~ perf_metrics = perf_metrics.groupby(['mouse', 'date']).apply(
        #~ lambda df: df.sort_values('n_trials').iloc[-1])
    

    ## Calculate first port poked
    # TODO: Why is this null in rare cases?
    # TODO: move this function to paclab repository
    big_trial_df['first_port_poked'] = big_poke_df[big_poke_df['first_poke']][
        'poked_port'].droplevel('poke')

    
    ## Convert port names to port dir
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

def parse_hdf5_files(path_to_terminal_data, mouse_names, 
    munged_sessions=None,
    rename_sessions_l=None,
    rename_mouse_d=None,
    protocol_name='PAFT'):
    """Load the data from the specified mice, clean, and return.
    
    path_to_terminal_data : string
        The path to Autopilot data. Can be gotten from
        paclab.paths.get_path_to_terminal_data()
    
    mouse_names : list of string
        A list of mouse names to load
    
    munged_sessions : list of string
        A list of session names to drop
    
    rename_sessions_l : list, or None
        If None, no renaming is done.
        Otherwise, each item in the list is a tuple of length 3
        Each tuple consists of (
            the session_name the file was saved as,
            the session_name it should have been saved as,
            the name of the actual mouse that was tested)
        Example:
            ('20220309115351-M2_PAFT-Box2', 
            '20220309115351-F2_PAFT-Box2', 
            'F2_PAFT',)        
    
    rename_mouse_d : dict or None
        If None, no renaming is done
        Otherwise, for each (key, value) pair, rename the mouse named
        "key" to "value".     

    protocol_name : string
        This is the protocol to load from the HDF5 file. 
    
    This function:
    * Uses load_data_from_all_mouse_hdf5 to load all mouse HDF5
    * Adds useful columns like t_wrt_start, rewarded_port, etc to pokes
    * Label the type of each poke as "correct", "error", "prev"
    * Label the outcome of the trial as the type of the first poke in that trial
    * Score sessions by fraction correct, rank of correct port
    * Calculated performance metrics over all sessions
    
    Returns: dict, with the following items
        'perf_metrics': DataFrame
            Performance metrics for each session
            Index: MultiIndex with levels mouse, date
            Columns: session_name, rcp, fc, n_trials, weight
    
        'session_df' : DataFrame
            Metadata about each session
            Index: MultiIndex with levels mouse, session_name
            Columns: box, date, orig_session_num, first_trial, last_trial,
                n_trials, approx_duration, weight
        
        'trial_data': DataFrame
            Metadata about each trial
            Index: MultiIndex with levels mouse, session_name, trial
            Columns: previously_rewarded_port, rewarded_port, timestamp_reward,
                trial_start, duration, n_pokes, correct, error, prev, 
                outcome, rcp
        
        'poke_data': DataFrame
            Metadata about each poke
            Index: MultiIndex with levels mouse, session_name, trial, poke
            Columns: poked_port, timestamp, trial_start, t_wrt_start, 
                rewarded_port, previously_rewarded_port, poke_type
    """
    ## Load trial data and weights from the HDF5 files
    # This also drops munged sessions
    session_df, trial_data, poke_data, sound_data = load_data_from_all_mouse_hdf5(
        mouse_names, munged_sessions=munged_sessions,
        path_to_terminal_data=path_to_terminal_data,
        rename_sessions_l=rename_sessions_l, 
        rename_mouse_d=rename_mouse_d,
        protocol_name=protocol_name)

    # Drop useless columns
    poke_data = poke_data.drop(['orig_session_num', 'date'], axis=1)
    trial_data = trial_data.drop('trial_num', axis=1)
    
    # This is a weights column that is no longer used
    session_df = session_df.drop('stop', axis=1, errors='ignore')

    # Rename
    trial_data = trial_data.rename(columns={'timestamp_trial_start': 'trial_start'})

    # Add duration
    trial_data['duration'] = (
        trial_data['trial_start'].shift(-1) - trial_data['trial_start']).apply(
        lambda ts: ts.total_seconds())


    ## Add columns to trial_data, and calculate t_wrt_start for pokes
    # Join 'trial_start' onto poke_data
    poke_data = poke_data.join(trial_data['trial_start'])

    # Drop pokes without a matching trial start
    # TODO: check that this only happens for the last trial in a session
    poke_data = poke_data[~poke_data['trial_start'].isnull()].copy()

    # Normalize poke time to trial start time
    poke_data['t_wrt_start'] = poke_data['timestamp'] - poke_data['trial_start']
    poke_data['t_wrt_start'] = poke_data['t_wrt_start'].apply(
        lambda ts: ts.total_seconds())


    ## Join rewarded_port and previously_rewarded_port on pokes
    poke_data = poke_data.join(
        trial_data[['rewarded_port', 'previously_rewarded_port']], 
        rsuffix='_correct')

    # Actually don't do this because then it looks like no pokes on this trial
    #~ # Drop poke_data rows where previously_rewarded_port is ''
    #~ # TODO: Check that this is only the first trial in a session
    #~ # TODO: first trial should be 0 not 1
    #~ poke_data = poke_data[poke_data['previously_rewarded_port'] != ''].copy()


    ## Label the type of each poke
    # target port is 'correct'
    # prev_target port is 'prev' (presumably consumptive)
    # all others (currently only one other type) is 'error'
    poke_data['poke_type'] = 'error'
    poke_data.loc[
        poke_data['poked_port'] == poke_data['rewarded_port'],
        'poke_type'] = 'correct'
    poke_data.loc[
        poke_data['poked_port'] == poke_data['previously_rewarded_port'],
        'poke_type'] = 'prev'


    ## Count pokes per trial and check there was at least 1 on every trial
    n_pokes = poke_data.groupby(
        ['mouse', 'session_name', 'trial']).size().rename('n_pokes')
    trial_data = trial_data.join(n_pokes)
    assert not trial_data['n_pokes'].isnull().any()
    assert (trial_data['n_pokes'] > 0).all()


    ## Debug there is always a correct poke
    n_poke_types_by_trial = poke_data.groupby(
        ['session_name', 'trial'])['poke_type'].value_counts().unstack(
        'poke_type').fillna(0).astype(int)
    assert len(trial_data) == len(n_poke_types_by_trial)
    assert (n_poke_types_by_trial['correct'] > 0).all()


    ## Label the outcome of each trial, based on the type of the first poke
    # time of first poke of each type
    first_poke = poke_data.reset_index().groupby(
        ['mouse', 'session_name', 'trial', 'poke_type']
        )['t_wrt_start'].min().unstack('poke_type')

    # Join
    trial_data = trial_data.join(first_poke)

    # Score by first poke (excluding prev)
    trial_outcome = trial_data[['correct', 'error']].idxmin(1)
    trial_data['outcome'] = trial_outcome


    ## Label trials by how many ports poked before correct
    # Get the latency to each port on each trial
    latency_by_port = poke_data.reset_index().groupby(
        ['mouse', 'session_name', 'trial', 'poked_port'])['t_wrt_start'].min()

    # Drop the consumption port (previous reward)
    consumption_port = trial_data[
        'previously_rewarded_port'].dropna().reset_index().rename(
        columns={'previously_rewarded_port': 'poked_port'})
    cp_idx = pandas.MultiIndex.from_frame(consumption_port)
    latency_by_port_dropped = latency_by_port.drop(cp_idx, errors='ignore')

    # Unstack the port onto columns
    lbpd_unstacked = latency_by_port_dropped.unstack('poked_port')

    # Rank them in order of poking
    # Subtract 1 because it starts with 1
    # The best is 0 (correct trial) and the worst is 6 (because consumption port
    # is ignored). The expectation under random choices is 3 (right??)
    lbpd_ranked = lbpd_unstacked.rank(
        method='first', axis=1).stack().astype(int) - 1

    # Find the rank of the correct port
    correct_port = trial_data['rewarded_port'].dropna().reset_index()
    cp_idx = pandas.MultiIndex.from_frame(correct_port)
    rank_of_correct_port = lbpd_ranked.reindex(
        cp_idx).droplevel('rewarded_port').rename('rcp')

    # Append this to big_trial_data
    trial_data = trial_data.join(rank_of_correct_port)

    # Error check
    assert not trial_data['rcp'].isnull().any()


    ## Score sessions by fraction correct
    scored_by_fraction_correct = trial_data.groupby(
        ['mouse', 'session_name'])[
        'outcome'].value_counts().unstack('outcome').fillna(0)
    scored_by_fraction_correct['perf'] = (
        scored_by_fraction_correct['correct'].divide(
        scored_by_fraction_correct.sum(axis=1)))


    ## Score sessions by n_trials
    scored_by_n_trials = trial_data.groupby(['mouse', 'session_name']).size()


    ## Score by n_ports
    scored_by_n_ports = trial_data.groupby(['mouse', 'session_name'])['rcp'].mean()


    ## Extract key performance metrics
    # This slices out sound-only trials
    perf_metrics = pandas.concat([
        scored_by_n_ports.rename('rcp'),
        scored_by_fraction_correct['perf'].rename('fc'),
        scored_by_n_trials.rename('n_trials'),
        ], axis=1, verify_integrity=True)

    # Join on weight and date (now date only)
    perf_metrics = perf_metrics.join(session_df[['date']])

    # Index by date
    perf_metrics = perf_metrics.reset_index().set_index(
        ['mouse', 'date']).sort_index()
    
    
    ## Return
    return {
        'session_df': session_df,
        'perf_metrics': perf_metrics,
        'trial_data': trial_data,
        'poke_data': poke_data,
        'sound_data': sound_data,
        }

def load_data_from_all_mouse_hdf5(mouse_names, munged_sessions,
    path_to_terminal_data, rename_sessions_l=None, rename_mouse_d=None,
    protocol_name='PAFT'):
    """Load trial data and weights from HDF5 files for all mice
    
    See load_data_from_single_hdf5 for how the data is loaded from each mouse.
    
    This function then concatenates the results over mice, drops the 
    sessions in `munged_sessions`, nullifies weights where they are saved 
    as 0, and error checks no more than one session per day.
    
    Some redundant and/or useless columns are dropped.
    
    Arguments:
        mouse_names : list
            A list of mouse names. Each should be an HDF5 file in 
            /home/chris/autopilot/data
        munged_sessions : list, or None
            A list of munged session names to drop.
            If None, drop no sessions
        rename_sessions_l : list or None
            If None, no renaming is done.
            Otherwise, each item in the list is a tuple of length 3
            Each tuple consists of (
                the session_name the file was saved as,
                the session_name it should have been saved as,
                the name of the actual mouse that was tested)
            Example:
                ('20220309115351-M2_PAFT-Box2', 
                '20220309115351-F2_PAFT-Box2', 
                'F2_PAFT',)
        rename_mouse_d : dict or None
            If None, no renaming is done
            Otherwise, for each (key, value) pair, rename the mouse named
            "key" to "value". 
        protocol_name : string
            The protocol name to load from the file
    
    Returns: session_df, trial_data
        session_df : DataFrame
            index : MultiIndex with levels 'mouse' and 'session_name'
                'mouse' : the values in `mouse_names`
                'session_name' : a string like '20210720133116-Male1_0720-Box1'
            columns:
                box : string, the box name
                orig_session_num : int, the original Autopilot session number
                first_trial : datetime, the timestamp of the first trial
                last_trial : datetime, the timestamp of the last trial
                n_trials : number of trials
                approx_duration : last_trial - first_trial
                date : datetime.date, the date of the session
                weight : float, the weight
    
        big_trial_data : DataFrame
            index : MultiIndex with levels mouse, session_name, and trial
                'mouse' : the values in `mouse_names`
                'session_name' : a string like '20210720133116-Male1_0720-Box1'
                'trial' : always starting with zero

            columns :
                'light' : True or False, whether a light was displayed
                'rpi' : One of ['rpi01' ... 'rpi08']
                'side' : One of ['L', 'R']
                'sound' : True or False, whether a sound was played
                'timestamp' : time of trial as a datetime
                    This is directly taken from the 'timestamp' columns in the HDF5
                    file, just decoded from bytes to datetime
                    So I think it is the "start" of the trial
    """
    # Iterate over provided mouse names
    msd_l = []
    mtd_l = []
    mpd_l = []
    mxd_l = []
    keys_l = []
    for mouse_name in mouse_names:
        # Form the hdf5 filename
        #~ h5_filename = '/home/chris/autopilot/data/{}.h5'.format(mouse_name)
        h5_filename = os.path.join(
            path_to_terminal_data, '{}.h5'.format(mouse_name))
        
        # Load data
        mouse_session_df, mouse_trial_data, mouse_poke_data, mouse_sound_data = (
            load_data_from_single_hdf5(mouse_name, h5_filename, protocol_name))
        
        # Skip if None
        if mouse_session_df is None and mouse_trial_data is None:
            continue
        else:
            assert mouse_session_df is not None
            assert mouse_trial_data is not None
            assert mouse_poke_data is not None
        
        # Store
        msd_l.append(mouse_session_df)
        mtd_l.append(mouse_trial_data)
        mpd_l.append(mouse_poke_data)
        mxd_l.append(mouse_sound_data)
        keys_l.append(mouse_name)
    
    # Concatenate
    session_df = pandas.concat(
        msd_l, keys=keys_l, names=['mouse'], verify_integrity=True)
    trial_data = pandas.concat(
        mtd_l, keys=keys_l, names=['mouse'], verify_integrity=True)
    poke_data = pandas.concat(
        mpd_l, keys=keys_l, names=['mouse'], verify_integrity=True)
    try:
        sound_data = pandas.concat(
            mxd_l, keys=keys_l, names=['mouse'], verify_integrity=True)
    except ValueError:
        sound_data = None

    # Drop munged sessions
    droppable_sessions = []
    if munged_sessions is not None:
        for munged_session in munged_sessions:
            if munged_session in session_df.index.levels[1]:
                droppable_sessions.append(munged_session)
            else:
                print("warning: cannot find {} to drop it".format(munged_session))
    if len(droppable_sessions) > 0:
        session_df = session_df.drop(droppable_sessions, level='session_name')
        trial_data = trial_data.drop(droppable_sessions, level='session_name')
        poke_data = poke_data.drop(droppable_sessions, level='session_name')
        
        if sound_data is not None:
            # Ignore errors here because not all sessions have sound_data
            sound_data = sound_data.drop(
                droppable_sessions, level='session_name', errors='ignore')


    ## Rename sessions that were saved by the wrong mouse name
    if rename_sessions_l is not None:
        # reset index
        trial_data = trial_data.reset_index()
        poke_data = poke_data.reset_index()
        session_df = session_df.reset_index()
        if sound_data is not None:
            sound_data = sound_data.reset_index()
        
        # fix
        for wrong_name, right_name, right_mouse in rename_sessions_l:
            # Fix trial_data
            bad_mask = trial_data['session_name'] == wrong_name
            trial_data.loc[bad_mask, 'session_name'] = right_name
            trial_data.loc[bad_mask, 'mouse'] = right_mouse

            # Fix poke_data
            bad_mask = poke_data['session_name'] == wrong_name
            poke_data.loc[bad_mask, 'session_name'] = right_name
            poke_data.loc[bad_mask, 'mouse'] = right_mouse            

            # Fix sound_data
            bad_mask = sound_data['session_name'] == wrong_name
            sound_data.loc[bad_mask, 'session_name'] = right_name
            sound_data.loc[bad_mask, 'mouse'] = right_mouse            

            # Fix session_df
            bad_mask = session_df['session_name'] == wrong_name
            session_df.loc[bad_mask, 'session_name'] = right_name
            session_df.loc[bad_mask, 'mouse'] = right_mouse

        # reset index back again
        trial_data = trial_data.set_index(
            ['mouse', 'session_name', 'trial']).sort_index()
        poke_data = poke_data.set_index(
            ['mouse', 'session_name', 'trial', 'poke']).sort_index()            
        sound_data = sound_data.set_index(
            ['mouse', 'session_name', 'sound']).sort_index()   
        session_df = session_df.set_index(
            ['mouse', 'session_name']).sort_index()


    ## Rename mice here
    if rename_mouse_d is not None:
        trial_data = trial_data.rename(
            index=rename_mouse_d, level='mouse').sort_index()
        poke_data = poke_data.rename(
            index=rename_mouse_d, level='mouse').sort_index()
        session_df = session_df.rename( 
            index=rename_mouse_d, level='mouse').sort_index()
        if sound_data is not None:
            sound_data = sound_data.rename(
                index=rename_mouse_d, level='mouse').sort_index()


    ## Error check only one session per mouse per day
    n_sessions_per_day = session_df.groupby(['mouse', 'date']).size()
    if not (n_sessions_per_day == 1).all():
        bad_ones = n_sessions_per_day.loc[n_sessions_per_day != 1].reset_index()
        print("warning: multiple sessions in {} case(s):\n{}".format(
            len(bad_ones), bad_ones))
        
        # First example
        bad_mouse = bad_ones['mouse'].iloc[0]
        bad_date = bad_ones['date'].iloc[0]
        bad_sessions = session_df.loc[
            session_df['date'] == bad_date].loc[bad_mouse].T
        print("remove one of these sessions:\n{}".format(bad_sessions))


    ## These weights columns are no longer used
    # Nullify zero weights
    #~ session_df.loc[session_df['weight'] < 1, 'weight'] = np.nan
    
    #~ # Drop useless columns from session_df
    #~ session_df = session_df.drop(
        #~ ['weights_date_string', 'weights_dt_start'], axis=1)
    
    
    ## Drop columns from trial_data that are redundant with session_df
    # (because this is how they were aligned)
    trial_data = trial_data.drop(
        ['orig_session_num', 'box', 'date'], axis=1)

    # Return
    return session_df, trial_data, poke_data, sound_data

def _parse_single_protocol_from_file(fi, this_protocol):
    """Load the trial data and continuous data from a single protocol
    
    This is only meant to be called from load_data_from_single_hdf5
    
    fi : open file handle
    this_protocol : name of protocol
    
    Returns: mouse_trial_data, mouse_poke_data, mouse_sound_data
    """
    # Load trial_data and weights from HDF5 file
    mouse_trial_data = pandas.DataFrame(
        fi.root['data'][this_protocol]['S00_PAFT']['trial_data'][:])
    
    # Get the list of all the sessions for which we have continuous data
    continuous_node = fi.root['data'][this_protocol]['S00_PAFT'][
        'continuous_data']
    continuous_session_names = list(continuous_node._v_children.keys())
    
    
    ## Get continuous_data from each session
    session_num = 1
    session_pokes_l = []
    session_pokes_keys_l = []
    session_sounds_l = []
    session_sounds_keys_l = []
    for session_name in continuous_session_names:
    
        ## Load this session
        session_node = continuous_node[session_name]
        session_num = int(session_name.split('_')[1])

        
        ## Load sounds (if any)
        if 'ChunkData_Sounds' in session_node:
            # This is only for the new version of the data (2022-06-29)
            sound_node = session_node['ChunkData_Sounds']
            session_sound_data = pandas.DataFrame.from_records(sound_node[:])
            
            # Store
            session_sounds_l.append(session_sound_data)
            session_sounds_keys_l.append(session_num)
    
        
        ## Load pokes
        if 'ChunkData_Pokes' in session_node and 'poked_port' in session_node:
            # This only happens when something has gone wrong
            # For instance, Subject deleted, then recreated, and now it
            # is storing old data and new data in the same place.
            # A workaround for now is just to load the old data
            # This is copied from "elif 'poked_port' in session_node" below
            print(
                "warning: old and new poke data mixed together in session"
                "{} in {}".format(session_num, fi.filename)
                )
            
            # Load poked_port and trial
            session_poke_port = pandas.DataFrame(session_node['poked_port'][:])
            session_poke_trial = pandas.DataFrame(session_node['trial'][:])

            # Stick them together
            # TODO: check timestamps match
            assert len(session_poke_port) == len(session_poke_trial)
            session_poke_port['trial'] = session_poke_trial['trial']

            # Store
            session_pokes_l.append(session_poke_port)
            session_pokes_keys_l.append(session_num)
        
        elif 'ChunkData_Pokes' in session_node:
            # This is the new version of the data (2022-06-29)
            poke_node = session_node['ChunkData_Pokes']
            session_poke_data = pandas.DataFrame.from_records(poke_node[:])
            
            # Store
            session_pokes_l.append(session_poke_data)
            session_pokes_keys_l.append(session_num)
            
        elif 'poked_port' in session_node:
            # This is the old version of the data (pre 2022-06-29)
            # Load poked_port and trial
            session_poke_port = pandas.DataFrame(session_node['poked_port'][:])
            session_poke_trial = pandas.DataFrame(session_node['trial'][:])

            # Stick them together
            # TODO: check timestamps match
            assert len(session_poke_port) == len(session_poke_trial)
            session_poke_port['trial'] = session_poke_trial['trial']

            # Store
            session_pokes_l.append(session_poke_port)
            session_pokes_keys_l.append(session_num)

        else:
            # Empty session with no continuous data, probably
            # something went wrong
            print(
                "munged continuous data in session "  
                "{} in {}".format(session_num, fi.filename)
                )

    # Concat pokes
    if len(session_pokes_l) == 0:
        mouse_poke_data = None
    else:
        mouse_poke_data = pandas.concat(
            session_pokes_l, keys=session_pokes_keys_l,
            names=['session', 'poke'])
    
    # Concat sounds
    if len(session_sounds_l) == 0:
        mouse_sound_data = None
    else:
        mouse_sound_data = pandas.concat(
            session_sounds_l, keys=session_sounds_keys_l,
            names=['session', 'sound'])    
    
    return mouse_trial_data, mouse_poke_data, mouse_sound_data

def load_data_from_single_hdf5(mouse_name, h5_filename, 
    protocol_name='PAFT'):
    """Load session and trial data from a single mouse's HDF5 file
    
    The trial data and the weights are loaded from the HDF5 file. The
    columns are decoded and coerced into more meaningful dtypes. Some
    useless columns are dropped. The box is inferred from the rpi names.
    
    The trial data and weights are aligned based on the original session
    number and the inferred box. This seems to work but is not guaranteed to 
    do so. A unique, sortable session_name is generated.
    
    The same session_names are in both returned DataFrames. 
    No sessions are dropped.
    
    Arguments:
        mouse_name : string
            Used to name the sessions
        h5_filename : string
            The filename to a single mouse's HDF5 file
        protocol_name : string
            The protocol to load from the file
    
    Returns: mouse_session_df, mouse_trial_data
        None, None if the hdf5 file can't be loaded
    
        Both of the following are sorted by their index.
        mouse_session_df : DataFrame
            index : string, the session name
                This is like 20210914135527-Male4_0720-Box2, based on the
                time of the first trial.
            columns: 
                box : string, the box name
                orig_session_num : int, the original Autopilot session number
                first_trial : datetime, the timestamp of the first trial
                last_trial : datetime, the timestamp of the last trial
                n_trials : number of trials
                approx_duration : last_trial - first_trial
                date : datetime.date, the date of the session
                weights_date_string : string, the date as a string use in the
                    weights field of the HDF5 file
                weight : float, the weight
                weights_dt_start : datetime, the datetime stored in the
                    weights field of the HDF5 file
        
        mouse_trial_data : DataFrame
            index : MultiIndex
                session_name : string, the unique session name
                trial : int, the trial number starting from 0
            columns :
                light, sound : bool, whether a  light or sound was played
                rpi : string, the name of the target rpi on that trial
                orig_session_num : int, the original Autopilot session number
                side : 'L' or 'R'
                timestamp : the trial time, directly from the HDF5 file
                box : 'Box1' or 'Box2', inferred from rpi name
                date : datetime.date, the date of the session
    """
    ## Load trial data and weights
    cannot_load = False
    try:
        with tables.open_file(h5_filename) as fi:
            # Get the weights
            mouse_weights = pandas.DataFrame(fi.root['history']['weights'][:])            

            # Either get all contained protocols or just the provided one
            if protocol_name is None:
                # Use all of them
                list_of_protocol_names = [
                    node._v_name for node in fi.root['data']._f_list_nodes()]
            else:
                # Just the provided one
                list_of_protocol_names = [protocol_name]
            
            # Iterate over protocol names and load each
            this_mouse_trial_data_l = []
            this_mouse_poke_data_l = []
            this_mouse_sound_data_l = []
            for this_protocol in list_of_protocol_names:
                # Load 
                loaded = _parse_single_protocol_from_file(fi, this_protocol)
                
                # Parse
                this_mouse_trial_data = loaded[0]
                this_mouse_poke_data = loaded[1]
                this_mouse_sound_data = loaded[2]
                
                # Store
                this_mouse_trial_data_l.append(this_mouse_trial_data)
                this_mouse_poke_data_l.append(this_mouse_poke_data)
                this_mouse_sound_data_l.append(this_mouse_sound_data)

    except tables.HDF5ExtError:
        cannot_load = True

    # Return None, None if loading error
    if cannot_load:
        print("cannot load {}".format(h5_filename))
        return None, None, None, None
    
    # Return None, None if no trials
    if np.sum(list(map(len, this_mouse_trial_data_l))) == 0:
        print("no trials to load from protocol {} in {}".format(
            protocol_name, h5_filename))
        return None, None, None, None
    
    # Return None, None if no protocols
    if len(list_of_protocol_names) == 0:
        print("no protocols to load in {}".format(h5_filename))
    
    # Concat the results over protocols
    # mouse_trial_data has a level on the index that is like trial number,
    # but I don't know if it's the same as the 'trial_num' or 'trial_in_session'
    # column, so leaving it unlabeled for now.
    mouse_trial_data = pandas.concat(
        this_mouse_trial_data_l, keys=list_of_protocol_names, 
        names=['protocol'])
    mouse_poke_data = pandas.concat(
        this_mouse_poke_data_l, keys=list_of_protocol_names, 
        names=['protocol'])
    try:
        # mouse_sound_data can be missing data from before this was
        # implemented, and that's okay
        mouse_sound_data = pandas.concat(
            this_mouse_sound_data_l, keys=list_of_protocol_names, 
            names=['protocol'])
        
        # Rename the column 'sound' which conflicts with index level 'sound'
        mouse_sound_data = mouse_sound_data.rename(
            columns={'sound': 'sound_type'})
        
    except ValueError:
        mouse_sound_data = None
    

    ## Sometimes trials have a blank timestamp_trial_start
    # I think this only happens when it crashes right away
    # Drop those trials, hopefully nothing else breaks
    mouse_trial_data = mouse_trial_data[
        mouse_trial_data['timestamp_trial_start'] != b''].copy()
    

    ## Coerce dtypes for mouse_trial_data
    # Decode columns that are bytes
    for decode_col in ['previously_rewarded_port', 'rewarded_port', 
            'timestamp_reward', 'timestamp_trial_start']:
        mouse_trial_data[decode_col] = (
            mouse_trial_data[decode_col].str.decode('utf-8')
            )

    # Coerce timestamp to datetime
    for timestamp_column in ['timestamp_reward', 'timestamp_trial_start']:
        mouse_trial_data[timestamp_column] = (
            mouse_trial_data[timestamp_column].apply(
            lambda s: datetime.datetime.fromisoformat(s)))

    # Coerce the columns that are boolean
    bool_cols = []
    for bool_col in bool_cols:
        mouse_trial_data[bool_col] = mouse_trial_data[bool_col].replace(
            {'True': True, 'False': False}).astype(bool)

    
    ## Coerce dtypes for mouse_poke_data
    # Decode columns that are bytes
    for decode_col in ['poked_port', 'timestamp']:
        mouse_poke_data[decode_col] = mouse_poke_data[
            decode_col].str.decode('utf-8')

    # Coerce timestamp to datetime
    for timestamp_column in ['timestamp']:
        mouse_poke_data[timestamp_column] = (
            mouse_poke_data[timestamp_column].apply(
            lambda s: datetime.datetime.fromisoformat(s)))

    # Coerce the columns that are boolean
    bool_cols = ['first_poke', 'reward_delivered']
    for bool_col in bool_cols:
        try:
            mouse_poke_data[bool_col] = mouse_poke_data[bool_col].replace(
                {1: True, 0: False}).astype(bool)
        except KeyError:
            print("warning: missing bool_col {}".format(bool_col))


    ## Coerce dtypes for mouse_sound_data
    if mouse_sound_data is not None:
        # Decode columns that are bytes
        for decode_col in ['pilot', 'side', 'sound_type', 'locking_timestamp']:
            mouse_sound_data[decode_col] = mouse_sound_data[
                decode_col].str.decode('utf-8')

        # Coerce timestamp to datetime
        for timestamp_column in ['locking_timestamp']:
            mouse_sound_data[timestamp_column] = (
                mouse_sound_data[timestamp_column].apply(
                lambda s: datetime.datetime.fromisoformat(s)))
    
    
    ## Coerce dtypes for mouse_weights
    # Rename more meaningfully
    mouse_weights = mouse_weights.rename(columns={'date': 'date_string'})
    
    # Decode
    mouse_weights['date_string'] = mouse_weights[
        'date_string'].str.decode('utf-8')

    # Convert 'date_string' to Timestamp
    mouse_weights['dt_start'] = mouse_weights['date_string'].apply(
        lambda s: datetime.datetime.strptime(s, '%y%m%d-%H%M%S'))
    
    # Calculate raw date (dropping time)
    mouse_weights['date'] = mouse_weights['dt_start'].apply(
        lambda dt: dt.date())
    
    
    ## Drop useless columns
    # Rename meaningfully
    mouse_weights = mouse_weights.rename(columns={
        'start': 'weight',
        'session': 'orig_session_num',
        })

    
    ## Asign 'box' based on 'rpi'
    # For mouse_trial_data
    mouse_trial_data['box'] = '???'
    box1_mask = mouse_trial_data['rewarded_port'].isin(
        ['rpi01', 'rpi02', 'rpi03', 'rpi04'])
    box2_mask = mouse_trial_data['rewarded_port'].isin(
        ['rpi05_L','rpi05_R','rpi06_L','rpi06_R','rpi07_L','rpi07_R','rpi08_L','rpi08_R'])
    box3_mask = mouse_trial_data['rewarded_port'].isin(
        ['rpi09_L','rpi09_R','rpi10_L','rpi10_R','rpi11_L','rpi11_R','rpi12_L','rpi12_R'])
    mouse_trial_data.loc[box1_mask, 'box'] = 'Box1'
    mouse_trial_data.loc[box2_mask, 'box'] = 'Box2'
    mouse_trial_data.loc[box3_mask, 'box'] = 'Box3'
    assert mouse_trial_data['box'].isin(['Box1', 'Box2', 'Box3']).all()

    # For mouse_poke_data
    mouse_poke_data['box'] = '???'
    box1_mask = mouse_poke_data['poked_port'].isin(
        ['rpi01', 'rpi02', 'rpi03', 'rpi04'])
    box2_mask = mouse_poke_data['poked_port'].isin(
        ['rpi05_L', 'rpi05_R', 'rpi06_L', 'rpi06_R', 'rpi07_L', 'rpi07_R', 'rpi08_L', 'rpi08_R'])
    box3_mask = mouse_poke_data['poked_port'].isin(
        ['rpi09_L','rpi09_R','rpi10_L','rpi10_R','rpi11_L','rpi11_R','rpi12_L','rpi12_R'])
    mouse_poke_data.loc[box1_mask, 'box'] = 'Box1'
    mouse_poke_data.loc[box2_mask, 'box'] = 'Box2'
    mouse_poke_data.loc[box3_mask, 'box'] = 'Box3'
    assert mouse_poke_data['box'].isin(['Box1', 'Box2', 'Box3']).all()
    

    ## Identify sessions
    # Sometimes the session number restarts numbering and I don't know why
    # Probably after every box change? Unclear
    # Session number definitely restarts after a protocol change
    # Let's group by ['box', 'protocol', 'date', 'orig_session_num'] and assume that 
    # each group represents a unique session. 
    # This could still fail if the renumbering happens without a box or
    # protocol change, somehow.
    # This will also fail if a session spans midnight

    # Rename to be explicit
    mouse_trial_data = mouse_trial_data.rename(
        columns={'session': 'orig_session_num'})
    
    # First add a date string
    mouse_trial_data['date'] = mouse_trial_data['timestamp_trial_start'].apply(
        lambda dt: dt.date())
    
    # Group by ['box', 'protocol', 'date', 'orig_session_num']
    # Assume each group is a unique session
    gobj = mouse_trial_data.groupby(
        ['box', 'protocol', 'date', 'orig_session_num'])
    
    # Extract times of first and last trials
    session_df = pandas.DataFrame.from_dict({
        'first_trial': gobj['timestamp_trial_start'].min(),
        'last_trial': gobj['timestamp_trial_start'].max(),
        'n_trials': gobj['timestamp_trial_start'].size(),
        })
    
    # Calculate approximate duration and do some basic sanity checks
    session_df['approx_duration'] = (
        session_df['last_trial'] - session_df['first_trial'])
    assert (
        session_df['approx_duration'] < datetime.timedelta(hours=2)).all()
    assert (
        session_df['approx_duration'] >= datetime.timedelta(seconds=0)).all()
    
    # Reset index
    session_df = session_df.reset_index()

    # Create a unique, sortable session name
    # This will generate a name like 20210907145607-Female4_0903-Box1
    # based on the time of the first trial
    session_df['session_name'] = session_df.apply(
        lambda row: '{}-{}-{}'.format(
        row['first_trial'].strftime('%Y%m%d%H%M%S'), mouse_name, row['box']),
        axis=1)
    
    
    ## Something is messed up with the weights for Bluefish_027 on 6-29
    ## Just don't do any of the weight stuff
    if False:
        ## Align session_df with mouse_weights
        # The assumption here is that each is unique based on 
        # ['date', 'orig_session_num']. Otherwise this won't work.
        assert not session_df[['date', 'orig_session_num']].duplicated().any()
        
        # Check for duplicates in mouse_weights
        dup_check = mouse_weights[['date', 'orig_session_num']].duplicated() 
        if dup_check.any():
            # Not sure why this is happening, must be some bug in the way it
            # was stored
            # Just drop the duplicates
            print("warning: duplicate sessions in {} mouse_weights on {}".format(
                mouse_name,
                ", ".join(map(str, mouse_weights.loc[dup_check, 'date'].values))
                ))
            mouse_weights = mouse_weights[~dup_check].copy()

        # Rename the weights columns to be more meaningful after merge
        mouse_weights = mouse_weights.rename(columns={
            'date_string': 'weights_date_string',
            'dt_start': 'weights_dt_start',
            })
        
        # Left merge, so we always have the same length as session_df
        # This drops extra rows in `mouse_weights`, corresponding to sessions
        # with no trials, typically after a munging event, which is fine
        session_df = pandas.merge(
            session_df, mouse_weights, how='left', 
            on=['date', 'orig_session_num'])
        
        # Make sure there was an entry in weights for every session
        # This is no longer true because of the dup_check drop above
        assert not session_df.isnull().any().any()


    ## Add the unique session_name to mouse_trial_data
    # Align on these columns which should uniquely defined
    join_on = ['box', 'protocol', 'date', 'orig_session_num']
    mouse_trial_data = mouse_trial_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    assert not mouse_trial_data['session_name'].isnull().any()


    ## Add the unique session_name to mouse_poke_data
    mouse_poke_data = mouse_poke_data.reset_index()
    
    # Get 'date' to align with session_df
    mouse_poke_data['date'] = mouse_poke_data['timestamp'].apply(
        lambda dt: dt.date())
    mouse_poke_data = mouse_poke_data.rename(
        columns={'session': 'orig_session_num'})
    
    # Align on these columns which should uniquely defined
    join_on = ['box', 'protocol', 'date', 'orig_session_num']
    mouse_poke_data = mouse_poke_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    
    # Drop any pokes here that are unaligned with sessions
    # This can happen if a session consisted only of one uncompleted trial
    to_drop = mouse_poke_data['session_name'].isnull()
    if to_drop.sum() > 0:
        pokes_to_drop = mouse_poke_data.loc[to_drop]
        dates_with_unaligned_pokes = ', '.join(
            map(str, pokes_to_drop['date'].unique()))
        print(
            "warning: dropping " 
            "{} unaligned pokes on dates {} from mouse {}".format(
            len(pokes_to_drop),
            dates_with_unaligned_pokes,
            mouse_name,            
            ))
        mouse_poke_data = mouse_poke_data.loc[~to_drop].copy()
    assert not mouse_poke_data['session_name'].isnull().any()

    
    ## Add the unique session_name to mouse_sound_data
    mouse_sound_data = mouse_sound_data.reset_index()
    
    # Get 'date' to align with session_df
    mouse_sound_data['date'] = mouse_sound_data['locking_timestamp'].apply(
        lambda dt: dt.date())
    mouse_sound_data = mouse_sound_data.rename(
        columns={'session': 'orig_session_num'})
    
    # Align on these columns which should uniquely defined
    join_on = ['protocol', 'date', 'orig_session_num']
    mouse_sound_data = mouse_sound_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    
    # Drop any sounds  here that are unaligned with sessions
    # This can happen if a session consisted only of one uncompleted trial
    to_drop = mouse_sound_data['session_name'].isnull()
    if to_drop.sum() > 0:
        sounds_to_drop = mouse_sound_data.loc[to_drop]
        dates_with_unaligned_sounds = ', '.join(
            map(str, sounds_to_drop['date'].unique()))
        print(
            "warning: dropping " 
            "{} unaligned sounds on dates {} from mouse {}".format(
            len(sounds_to_drop),
            dates_with_unaligned_sounds,
            mouse_name,
            ))
        mouse_sound_data = mouse_sound_data.loc[~to_drop].copy()
    assert not mouse_sound_data['session_name'].isnull().any()
    
    
    ## Index everything by session_name and whatever else makes sense
    mouse_trial_data = mouse_trial_data.rename(
        columns={'trial_in_session': 'trial'})    
    
    # Identify any duplicated trials, and drop those sessions
    # Not sure why this happens!
    dups = mouse_trial_data.loc[
        mouse_trial_data.duplicated(subset=['session_name', 'trial'], 
        keep=False)][['session_name', 'trial']]
    if len(dups) > 0:
        dup_sessions = dups['session_name'].unique()
        print("warning: dropping sessions with duplicated trials:\n{}".format(
            str(dups)))
        mouse_trial_data = mouse_trial_data[
            ~mouse_trial_data['session_name'].isin(dup_sessions)].copy()
    
    # Set index
    # This is where we drop the unlabeled level of the index that was
    # kind of like 'trial', replacing it with the more explicit 'trial'
    mouse_trial_data = mouse_trial_data.reset_index('protocol').set_index(
        ['session_name', 'trial'])
    mouse_poke_data = mouse_poke_data.set_index(
        ['session_name', 'trial', 'poke'])
    mouse_sound_data = mouse_sound_data.set_index(
        ['session_name', 'sound'])        
    session_df = session_df.set_index('session_name')
    
    # Sort index
    mouse_trial_data = mouse_trial_data.sort_index()
    mouse_poke_data = mouse_poke_data.sort_index()
    mouse_sound_data = mouse_sound_data.sort_index()
    session_df = session_df.sort_index()

    # Error check
    assert not session_df.index.duplicated().any()
    assert not mouse_trial_data.duplicated().any()
    assert not mouse_sound_data.duplicated().any()
    assert not mouse_poke_data.duplicated().any()
    
    
    ## Return
    return session_df, mouse_trial_data, mouse_poke_data, mouse_sound_data

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
        'poked_port',
        ]

    # Do each column
    for replace_col in replace_cols:
        if replace_col in res.columns:
            # Replace
            res[replace_col + '_dir'] = res[replace_col].replace(replacing_d)

    return res