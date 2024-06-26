## For parsing files from Summer 2022 (before sandboxes)
# This module is not imported by default

# We reuse some of the same helper functions
from .parse import *

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

    
    ## Before 2022-06-29, the first trial was often 1 instead of 0
    # The other clue is that previously_rewarded_port was '' in those cases
    # Fix this
    subdf_l = []
    this_poke_data_l = []
    this_poke_data_keys_l = []
    for keys, subdf in trial_data.groupby(['mouse', 'session_name']):
        # Get trial data for this session
        subdf = subdf.reset_index()
        
        # Get corresponding pokedata
        this_poke_data = poke_data.loc[keys].reset_index()
        
        if (
                subdf['trial'].iloc[0] == 1 and 
                (subdf['previously_rewarded_port'].iloc[0] == '')):
            # Relabel trial count
            subdf['trial'] = subdf['trial'] - 1
            
            # Do the same in poke_data
            this_poke_data['trial'] = this_poke_data['trial'] - 1
        
        # Store
        subdf = subdf.set_index(['mouse', 'session_name', 'trial'])
        subdf_l.append(subdf)
        this_poke_data_l.append(this_poke_data.set_index('poke'))
        this_poke_data_keys_l.append(keys)
    
    # Concat
    trial_data = pandas.concat(subdf_l).sort_index()
    poke_data = pandas.concat(
        this_poke_data_l, keys=this_poke_data_keys_l, 
        names=['mouse', 'session_name']).sort_index()
    
    
    ## Clean trial data
    # Rename
    trial_data = trial_data.rename(
        columns={'timestamp_trial_start': 'trial_start'})

    # Add duration
    trial_data['duration'] = (
        trial_data['trial_start'].shift(-1) - trial_data['trial_start']).apply(
        lambda ts: ts.total_seconds())

    
    ## From here on out, this is the same as the newer parsing code
    ## Clean poke data
    # Drops pokes from before the first trial, after last trial, and those
    # that can't be aligned to any trial start
    # Calculates t_wrt_start for each poke    
    txt_output = ''
    poke_data, txt_output = clean_big_poke_df(
        poke_data, trial_data, txt_output, False)


    ## Label poke type and identify trials with no choice made
    # Sort properly by timestamp (old poke idx is now poke_orig_idx)
    poke_data = reorder_pokes_by_timestamp(poke_data)

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
    poke_data, trial_data = label_poke_types(poke_data, trial_data)


    # Sets 'poke_rank' in big_poke_df and 'rcp', 'outcome', and 
    # 'first_port_poked' in big_trial_df
    # 'rcp' is null on all 'no_correct_pokes'
    # 'first_port_poked' is null and 'outcome' is 'spoiled' on all 'no_choice_made'
    poke_data, trial_data = label_trial_outcome(poke_data, trial_data)

    # Warn about trials with these errors
    # TODO: drop them?
    txt_output = warn_about_munged_trials(trial_data, txt_output, False)

    # Convert port names to port dir and calculate err_dist
    poke_data, trial_data = calculate_distance_between_choice_ports(
        poke_data, trial_data)
    
    
    ## Score sessions into perf_metrics
    perf_metrics = calculate_perf_metrics(trial_data)

    # Join date
    perf_metrics = perf_metrics.join(
        session_df['date'], on=['mouse', 'session_name'])

    # Join n_session for each mouse
    session_df['n_session'] = -1
    for mouse, subdf in session_df.groupby('mouse'):
        ranked = subdf['first_trial'].rank(method='first').astype(int) - 1
        session_df.loc[ranked.index, 'n_session'] = ranked.values
    assert not (session_df['n_session'] == -1).any()

    # Join n_session on perf_metrics
    perf_metrics = perf_metrics.join(session_df['n_session'])
    
    
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
    ## This is used for all timestamps
    tz = pytz.timezone('America/New_York')
    
    
    ## Load trial data and weights
    cannot_load = False
    try:
        with tables.open_file(h5_filename) as fi:
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
    mouse_trial_data, mouse_poke_data, mouse_sound_data = (
        decode_and_coerce_all_df(
            mouse_trial_data, 
            mouse_poke_data, 
            mouse_sound_data, 
            tz, 
            old_data=True))


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


