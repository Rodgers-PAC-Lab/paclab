"""For parsing HDF5 files and returning DataFrames"""
import datetime
import os
import numpy as np
import tables
import pandas

def parse_hdf5_files(path_to_terminal_data, mouse_names, rename_sessions_l=None):
    """Load the data from the specified mice, clean, and return.
    
    path_to_terminal_data : string
        The path to Autopilot data. Can be gotten from
        paclab.paths.get_path_to_terminal_data()
    
    mouse_names : list of string
        A list of mouse names to load
    
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
    session_df, trial_data, poke_data = load_data_from_all_mouse_hdf5(
        mouse_names, munged_sessions=[],
        path_to_terminal_data=path_to_terminal_data,
        rename_sessions_l=rename_sessions_l)

    # Drop useless columns
    poke_data = poke_data.drop(['orig_session_num', 'date'], axis=1)
    trial_data = trial_data.drop('trial_num', axis=1)
    session_df = session_df.drop('stop', axis=1)

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
        'outcome'].value_counts().unstack('outcome')
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

    # Join on weight and date
    perf_metrics = perf_metrics.join(session_df[['date', 'weight']])

    # Index by date
    perf_metrics = perf_metrics.reset_index().set_index(
        ['mouse', 'date']).sort_index()
    
    
    ## Return
    return {
        'session_df': session_df,
        'perf_metrics': perf_metrics,
        'trial_data': trial_data,
        'poke_data': poke_data,
        }

def load_data_from_all_mouse_hdf5(mouse_names, munged_sessions,
    path_to_terminal_data, rename_sessions_l=None):
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
        munged_sessions : list
            A list of munged session names to drop.
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
    keys_l = []
    for mouse_name in mouse_names:
        # Form the hdf5 filename
        #~ h5_filename = '/home/chris/autopilot/data/{}.h5'.format(mouse_name)
        h5_filename = os.path.join(
            path_to_terminal_data, '{}.h5'.format(mouse_name))
        
        # Load data
        mouse_session_df, mouse_trial_data, mouse_poke_data = (
            load_data_from_single_hdf5(mouse_name, h5_filename))
        
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
        keys_l.append(mouse_name)
    
    # Concatenate
    session_df = pandas.concat(msd_l, keys=keys_l, names=['mouse'])
    trial_data = pandas.concat(mtd_l, keys=keys_l, names=['mouse'])
    poke_data = pandas.concat(mpd_l, keys=keys_l, names=['mouse'])

    # Drop munged sessions
    droppable_sessions = []
    for munged_session in munged_sessions:
        if munged_session in session_df.index.levels[1]:
            droppable_sessions.append(munged_session)
        else:
            print("warning: cannot find {} to drop it".format(munged_session))
    session_df = session_df.drop(droppable_sessions, level='session_name')
    trial_data = trial_data.drop(droppable_sessions, level='session_name')
    poke_data = poke_data.drop(droppable_sessions, level='session_name')


    ## Rename sessions that were saved by the wrong mouse name
    if rename_sessions_l is not None:
        # reset index
        trial_data = trial_data.reset_index()
        poke_data = poke_data.reset_index()
        session_df = session_df.reset_index()
        
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

            # Fix session_df
            bad_mask = session_df['session_name'] == wrong_name
            session_df.loc[bad_mask, 'session_name'] = right_name
            session_df.loc[bad_mask, 'mouse'] = right_mouse

        # reset index back again
        trial_data = trial_data.set_index(
            ['mouse', 'session_name', 'trial']).sort_index()
        poke_data = poke_data.set_index(
            ['mouse', 'session_name', 'trial', 'poke']).sort_index()            
        session_df = session_df.set_index(
            ['mouse', 'session_name']).sort_index()


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

    # Nullify zero weights
    session_df.loc[session_df['weight'] < 1, 'weight'] = np.nan
    
    # Drop useless columns from session_df
    session_df = session_df.drop(
        ['weights_date_string', 'weights_dt_start'], axis=1)
    
    # Drop columns from trial_data that are redundant with session_df
    # (because this is how they were aligned)
    trial_data = trial_data.drop(
        ['orig_session_num', 'box', 'date'], axis=1)

    # Return
    return session_df, trial_data, poke_data

def load_data_from_single_hdf5(mouse_name, h5_filename):
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
            mouse_trial_data = pandas.DataFrame(
                fi.root['data']['PAFT']['S00_PAFT']['trial_data'][:])
            mouse_weights = pandas.DataFrame(
                fi.root['history']['weights'][:])
            
            # get sessions one at a time
            session_num = 1
            session_pokes_l = []
            session_pokes_keys_l = []
            while True:
                # Load this session if it exists
                try:
                    session_node = (
                        fi.root['data']['PAFT']['S00_PAFT']['continuous_data'][
                        'session_{}'.format(session_num)])
                except IndexError:
                    break
                
                if 'poked_port' in session_node:
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
                    pass

                # Iterate
                session_num += 1
            
            # Concat pokes
            mouse_poke_data = pandas.concat(
                session_pokes_l, keys=session_pokes_keys_l, 
                names=['session', 'poke'])

    except tables.HDF5ExtError:
        cannot_load = True
    
    if cannot_load:
        print("cannot load {}".format(h5_filename))
        return None, None
    

    ## Coerce dtypes for mouse_trial_data
    # Decode columns that are bytes
    for decode_col in ['previously_rewarded_port', 'rewarded_port', 
            'timestamp_reward', 'timestamp_trial_start']:
        mouse_trial_data[decode_col] = mouse_trial_data[decode_col].str.decode('utf-8')

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
    bool_cols = []
    for bool_col in bool_cols:
        mouse_poke_data[bool_col] = mouse_poke_data[bool_col].replace(
            {'True': True, 'False': False}).astype(bool)
    
    
    ## Coerce dtypes for mouse_weights
    # Rename more meaningfully
    mouse_weights = mouse_weights.rename(columns={'date': 'date_string'})
    
    # Decode
    mouse_weights['date_string'] = mouse_weights['date_string'].str.decode('utf-8')

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
    # Let's group by ['box', 'date', 'session'] and assume that each
    # is unique. This could still fail if the renumbering happens without
    # a box change, somehow.
    # This will also fail if a session spans midnight
    
    # First add a date string
    mouse_trial_data['date'] = mouse_trial_data['timestamp_trial_start'].apply(
        lambda dt: dt.date())
    
    # Group by ['box', 'date', 'session']
    gobj = mouse_trial_data.groupby(['box', 'date', 'session'])
    
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
    
    # Reset index, and preserve the original session number
    session_df = session_df.reset_index()
    session_df = session_df.rename(columns={'session': 'orig_session_num'})

    # Extract the date of the session
    session_df['date'] = session_df['first_trial'].apply(
        lambda dt: dt.date())

    # Create a unique, sortable session name
    # This will generate a name like 20210907145607-Female4_0903-Box1
    # based on the time of the first trial
    session_df['session_name'] = session_df.apply(
        lambda row: '{}-{}-{}'.format(
        row['first_trial'].strftime('%Y%m%d%H%M%S'), mouse_name, row['box']),
        axis=1)
    
    
    ## Align session_df with mouse_weights
    # The assumption here is that each is unique based on 
    # ['date', 'orig_session_num']. Otherwise this won't work.
    assert not session_df[['date', 'orig_session_num']].duplicated().any()
    assert not mouse_weights[['date', 'orig_session_num']].duplicated().any()

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
    assert not session_df.isnull().any().any()


    ## Add the unique session_name to mouse_trial_data
    # Get 'date' to align with session_df
    # This will fail if the session crossed over midnight
    mouse_trial_data['date'] = mouse_trial_data['timestamp_trial_start'].apply(
        lambda dt: dt.date())
    mouse_trial_data = mouse_trial_data.rename(
        columns={'session': 'orig_session_num'})
    
    # Align on these columns which should uniquely defined
    join_on = ['box', 'date', 'orig_session_num']
    mouse_trial_data = mouse_trial_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    assert not mouse_trial_data['session_name'].isnull().any()
    

    ## Add the unique session_name to mouse_poke_data
    mouse_poke_data = mouse_poke_data.reset_index()
    
    # Get 'date' to align with session_df
    # This will fail if the session crossed over midnight
    mouse_poke_data['date'] = mouse_poke_data['timestamp'].apply(
        lambda dt: dt.date())
    mouse_poke_data = mouse_poke_data.rename(
        columns={'session': 'orig_session_num'})
    
    # Align on these columns which should uniquely defined
    join_on = ['box', 'date', 'orig_session_num']
    mouse_poke_data = mouse_poke_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    assert not mouse_poke_data['session_name'].isnull().any()
    
    
    ## Index everything by session_name and whatever else makes sense
    mouse_trial_data = mouse_trial_data.rename(
        columns={'trial_in_session': 'trial'})    
    mouse_trial_data = mouse_trial_data.set_index(
        ['session_name', 'trial'])
    mouse_poke_data = mouse_poke_data.set_index(
        ['session_name', 'trial', 'poke'])
    session_df = session_df.set_index('session_name')
    
    mouse_trial_data = mouse_trial_data.sort_index()
    mouse_poke_data = mouse_poke_data.sort_index()
    session_df = session_df.sort_index()

    # Error check
    assert not session_df.index.duplicated().any()

    
    ## Return
    return session_df, mouse_trial_data, mouse_poke_data
