"""This module contains functions for parsing octopilot files.

load_session : Load data from a particular session, given a session name
"""
import os
import glob
import datetime
import scipy.stats
import numpy as np
import pytz
import pandas
import warnings
import ast

def str2dt(s):
    """Transform string `s` into a datetime in timezone `America/New_York`"""
    tz = pytz.timezone('America/New_York')
    return datetime.datetime.fromisoformat(s).astimezone(tz)

def choose_sandboxes_to_enter(
    sandbox_root_dir='~/mnt/cuttlefish/behavior/from_clownfish/octopilot/logs', 
    include_sessions=None, 
    mouse_names=None, 
    munged_sessions=None,
    ):
    """Identify which sandboxes to enter for a given mouse or session list
    
    sandbox_root_dir : str
        Path to the root directory containing octopilot logs
        Within `sandbox_root_dir` should be a folder for each year, and
        within that a folder for each month, and within that a folder for
        each session (a "sandbox").
        The sandbox name should be '{dt_string}_{mouse_name}' where
        `dt_string` is a 19-character string like '2024-10-11_17-26-20'
    
    include_sessions, mouse_names, munged_sessions :
        passed directly from parse_sandboxes, see documentation there
    
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
    # Allow a tilde in `sandbox_root_dir`
    sandbox_root_dir = os.path.expanduser(sandbox_root_dir)
    
    # Select only year directories (guards against expensive expansion)
    year_directories = [
        s for s in os.listdir(sandbox_root_dir) 
        if len(s) == 4 and s.startswith('20')]
    
    # Error check that there's at least one year directory (guards against
    # incorrect sandbox root dir)
    if len(year_directories) == 0:
        raise IOError(f'no year subdirectories found in {sandbox_root_dir}')
    
    # Find all potential sandbox directories ('root/year/month/session/')
    # The directory must contain trials.csv, which avoids certain edge 
    # cases like a crash after creating the sandbox or logfile
    sandbox_dir_l = []
    for year in year_directories:
        sandbox_dir_l += glob.glob(
            os.path.join(sandbox_root_dir, year, '*/*/trials.csv'))
    
    # Abspath and remove the final trials.csv
    def fix(pth):
        return os.path.split(os.path.abspath(pth))[0]
    sandbox_dir_l = sorted(map(fix, sandbox_dir_l))
    
    # Form DataFrame
    sandbox_df = pandas.Series(sandbox_dir_l).rename('full_path').to_frame()
    
    # Extract sandbox name
    # This works because of abspath above
    sandbox_df['sandbox_name'] = sandbox_df['full_path'].apply(
        lambda s: os.path.split(s)[1])
    
    # Extract sandbox_dt_string and mouse name
    # This works because dt_string is always 19 chars long
    sandbox_df['sandbox_dt_string'] = sandbox_df['sandbox_name'].str[:19]
    sandbox_df['mouse_name'] = sandbox_df['sandbox_name'].str[20:]
    
    # Error checks
    assert not (sandbox_df['mouse_name'] == '').any()
    assert not sandbox_df['mouse_name'].isnull().any()
    
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

def load_session(
    octopilot_root='~/mnt/cuttlefish/behavior/from_clownfish/octopilot/logs', 
    octopilot_session_name=None, 
    suppress_order_warnings=False,
    ):
    """Load data from octopilot session
    
    octopilot_root : str
        Path to root of octopilot data
    
    octopilot_session_name : str
    
    suppress_order_warnings : bool
        If True, suppresses out-of-order warnings in sync_sounds
    
    Workflow
    * Load the CSV files from the octopilot session.
      This step dominates the running time.
    * Coerce datetimes, rename some columns
    * Compute sounds['speaker_frame'], the frame on which sound came out
      of the speaker.
    * Compute the linear relationship between sounds['speaker_frame'] and 
      wall time on each pi. This should be nearly perfect because it's just
      a match between clocks on the same device. The sound time is stored
      as 'speaker_time_s'
    * Line up n_sound between sounds and sound_plans
    * Reshapes the flash data.
    
    Returns : dict
        'sounds': DataFrame or None
            This will be None in the case that the file was empty.
            The individual frames of audio that were sent to hifiberry.
            Columns:
            'trial_number', 'rpi', 'n_sound', 'message_time', 'data_left',
            'data_right', 'data_hash', 'last_frame_time',
            'frames_since_cycle_start', 'message_time_s', 'message_frame',
            'speaker_frame', 'speaker_time_s', 'diff_speaker_time_s', 'side',
            'gap_chunks'   
        
        'sound_plans': DataFrame or None
            This will be None in the case that the file did not exist
            The noise bursts that were generated by octopilot. Not all 
            were actually played, and some were played multiple times, depending
            on trial duration.
            Columns: time, side, gap, gap_chunks
            Index: trial_number * rpi * n_sound
        
        'trials': DataFrame
            One row per trial.
            Index: trial_number
            Columns: 'start_time', 'start_time_s', 'goal_port', 'trigger',
               'reward_time'
            Other columns may be included depending on the stimulus set.
            
        'pokes': DataFrame
            One row per poke.
            Columns: poke_time, trial_number, rpi, poked_port, rewarded
        
        'flashes': DataFrame
            Index: trial_number * rpi
            Columns: dt, relative
            'dt': The datetime at which each rpi flashed
            'relative': (float) number of seconds since session start
            Session start is defined as trials['start_time'].iloc[0]
        
        'box_params': dict with keys
            'zmq_port', 'bonsaid_port', 'desktop_ip', 'bonsai_ip',
            'ypos_of_gui', 'camera', 'name'
        
        'task_params': dict with keys
            'name', and whatever else is in the task_params.json
        
        'mouse_params': dict with keys
            'reward_value', 'box', 'task', 'name'
        
        'connected_pis': DataFrame
            This information comes from box_params but is popped out to 
            avoid nesting.
            Index: 'name' which is the rpi
            Columns: 'ip', 'left_port_position', 'right_port_position',
                'left_port_name', 'right_port_name'
    """
    ## Error check
    if octopilot_session_name is None:
        raise ArgumentError('octopilot session name cannot be None')
    
    
    ## Form session dir
    octopilot_year, octopilot_month = octopilot_session_name.split('-')[:2]
    octopilot_session_dir = os.path.join(
        os.path.expanduser(octopilot_root), 
        octopilot_year, 
        octopilot_month, 
        octopilot_session_name)

    
    ## Load config from logfile
    # Ideally we would be storing box_params, task_params, mouse_params as files
    # in the directory. Because we're not, grab them from the logfile instead. 
    # Load the first bit of the logfile
    logfile_path = os.path.join(octopilot_session_dir, 'logfile')
    with open(logfile_path) as fi:
        lines = fi.readlines(4000) # bytes, approx
    
    # The first 1-2 lines are INFO lines, possibly identifying the camera
    line_contains_info = [line.startswith('[INFO]') for line in lines]
    n_info_lines = np.sum(line_contains_info[:5])
    
    # Parse camera_name if it was provided
    # This turns out to be unnecessary since the camera in box_params is more
    # likely to be accurate (not always specified here)
    if n_info_lines == 1:
        assert lines[0] == '[INFO] - Initializing Dispatcher with the following params:\n'
        camera_name = None
    
    elif n_info_lines == 2:
        if lines[0] == '[INFO] - warning: no camera specified\n':
            camera_name = None
        
        elif lines[0].startswith('[INFO] - will use camera '):
            camera_name = lines[0].strip()[25:]
        
        else:
            raise ValueError(
                f'malformed line 0 in {logfile_path}: {lines[0].strip()}')
    
    else:
        raise ValueError(f'unusual number of info lines in {logfile_path}')
    
    # The next three lines encode box, task, and mouse params
    # Note that there is a trailing semicolon to drop
    # Parse the box params
    assert lines[n_info_lines].startswith('box_params: ')
    box_params_s = lines[n_info_lines].strip()[12:-1]
    box_params = ast.literal_eval(box_params_s)
    
    # Parse the task params
    assert lines[n_info_lines + 1].startswith('task_params: ')
    task_params_s = lines[n_info_lines + 1].strip()[13:-1]
    task_params = ast.literal_eval(task_params_s)
    
    # Parse the mouse params
    assert lines[n_info_lines + 2].startswith('mouse_params: ')
    mouse_params_s = lines[n_info_lines + 2].strip()[14:-1]
    mouse_params = ast.literal_eval(mouse_params_s)    

    # In some cases box_params['camera'] is None and sometimes ''
    # Make it uniformly None
    if 'camera' in box_params and box_params['camera'] == '':
        box_params['camera'] = None
    

    ## Load behavior data
    # Allow "fixing" - if trials.csv.fixed exists, load that one instead
    if os.path.exists(os.path.join(octopilot_session_dir, 'trials.csv.fixed')):
        print(
            f'warning: {octopilot_session_name} filename ending in *.fixed '
            'found, using that one')
        sounds = pandas.read_table(
            os.path.join(octopilot_session_dir, 'sounds.csv.fixed'), sep=',')
        sound_plans = pandas.read_table(
            os.path.join(octopilot_session_dir, 'sound_plans.csv'), sep=',')
        trials = pandas.read_table(
            os.path.join(octopilot_session_dir, 'trials.csv.fixed'), sep=',')
        pokes = pandas.read_table(
            os.path.join(octopilot_session_dir, 'pokes.csv.fixed'), sep=',')
        flashes = pandas.read_table(
            os.path.join(octopilot_session_dir, 'flashes.csv'), sep=',', 
            header=None)
    
    else:
        sounds = pandas.read_table(
            os.path.join(octopilot_session_dir, 'sounds.csv'), sep=',')
        
        # In early 2025, this was not saved
        try:
            sound_plans = pandas.read_table(
                os.path.join(octopilot_session_dir, 'sound_plans.csv'), sep=',')
        except FileNotFoundError:
            sound_plans = None

        trials = pandas.read_table(
            os.path.join(octopilot_session_dir, 'trials.csv'), sep=',')
        pokes = pandas.read_table(
            os.path.join(octopilot_session_dir, 'pokes.csv'), sep=',')

        # This file now seems to always exist, so this try/except is optional
        try:
            flashes = pandas.read_table(
                os.path.join(octopilot_session_dir, 'flashes.csv'), sep=',', 
                header=None)
        except FileNotFoundError:
            flashes = None
    
    # Label the index name
    pokes.index.name = 'poke'
    if sound_plans is not None:
        sound_plans.index.name = 'plan'
    if flashes is not None:
        flashes.index.name = 'flash'

    # Rename this one to indicate that it is the time the message was
    # sent about the sound, which is definitely not the time it played
    # TODO: do this in octopilot itself
    sounds = sounds.rename(columns={'sound_time': 'message_time'})

    # TODO: fix this missing header
    if flashes is not None:
        flashes.columns = ['trial_number', 'rpi', 'flash_time']

    # Index trials by trial_number
    trials = trials.set_index('trial_number')

    # Coerce
    # This format='ISO8601' syntax is because isoformat() sometimes rounds off
    # the microseconds, producing inconsistent formats in the input text
    tz = pytz.timezone('America/New_York')
    if flashes is not None:
        flashes['flash_time'] = pandas.to_datetime(
            flashes['flash_time'], format='ISO8601',
            ).dt.tz_localize(tz)
    sounds['message_time'] = pandas.to_datetime(
        sounds['message_time'], format='ISO8601',
        ).dt.tz_localize(tz)
    pokes['poke_time'] = pandas.to_datetime(
        pokes['poke_time'], format='ISO8601',
        ).dt.tz_localize(tz)
    trials['start_time'] = pandas.to_datetime(
        trials['start_time'], format='ISO8601',
        ).dt.tz_localize(tz)
    trials['reward_time'] = pandas.to_datetime(
        trials['reward_time'], format='ISO8601',
        ).dt.tz_localize(tz)


    ## Define session_start_time
    if len(trials) >= 1:
        # Take the time of the first trial on the desktop
        # It doesn't really matter whether this is the same time on all of the
        # other pis. This is just a point of reference to convert datetimes to floats
        session_start_time = trials['start_time'].iloc[0]

        # Define time in session
        trials['start_time_s'] = (
            trials['start_time'] - session_start_time).dt.total_seconds()

        # Convert datetimes to time in seconds since session_start_time
        sounds['message_time_s'] = (
            sounds['message_time'] - session_start_time).dt.total_seconds()

    else:
        print(f'warning: session {octopilot_session_name} has zero trials')

    
    ## Define flash_wrt_session_start
    if flashes is not None:
        # Error check no duplicates
        if (flashes.groupby(['trial_number', 'rpi']).size() > 1).any():
            raise ValueError(
                f'non-unique flashes in {octopilot_session_name}')
        
        # Put rpi on columns of flashes
        flashes = flashes.set_index(
            ['trial_number', 'rpi'])['flash_time'].unstack('rpi')

        # The start times according to the desktop
        trial_start_times_clock = trials['start_time']

        # The flash times on each rpi, wrt trial_start_time
        # This is never negative, peaks at 10ms, generally <20ms, 
        # but very long tail out to 100ms
        flash_wrt_trial_start = flashes.sub(
            trial_start_times_clock, axis=0).apply(
            lambda ser: ser.dt.total_seconds())

        # The flash times on each rpi, wrt session_start_time
        flash_wrt_session_start = flashes.sub(
            session_start_time).apply(
            lambda ser: ser.dt.total_seconds())

        # Concat these two similar ones
        flashes = pandas.concat(
            [flashes, flash_wrt_session_start], 
            axis=1, keys=['dt', 'relative'], names=['typ'])

        # Stack because the columns differ across boxes
        flashes = flashes.stack(future_stack=True)


    ## Sync the sounds and the sound plans
    # TODO: handle the case where sounds is empty (should not be happening
    # that often, but did in early 2025). An edge case is when there are
    # some sounds but not enough to correctly calculate the sync between
    # rpi clock and desktop clock
    if len(sounds) > 0:
        # This call is a little slow, but most of the running time is
        # dominated by reading the text files above
        sounds = sync_sounds(
            sounds, octopilot_session_name, 
            suppress_order_warnings=suppress_order_warnings)

        # Join sound plans onto sounds
        if sound_plans is not None:
            sounds, sound_plans = join_sound_plans_on_sounds(
                sounds, sound_plans)

        else:
            print(f'warning: no sound plan in {session_name}')

    else:
        sounds = None
    

    ## Pop this dict out
    connected_pis = box_params.pop('connected_pis')
    connected_pis = pandas.DataFrame(connected_pis).set_index('name')
    

    ## Return
    return {
        'sounds': sounds,
        'sound_plans': sound_plans,
        'trials': trials,
        'pokes': pokes,
        'flashes': flashes,
        'box_params': box_params,
        'mouse_params': mouse_params,
        'task_params': task_params,
        'connected_pis': connected_pis,
        }

def sync_sounds(sounds, octopilot_session_name, suppress_order_warnings=False):
    """Estimates the clock time of each row in `sounds`.
    
    For each frame of audio, the rpi reports the current clock time and
    jack audio frame. We can use that to compute the relationship between
    jack audio frames and clock time. Then, after accounting for buffer delays,
    we can estimate the time that the sound was played.
    
    Workflow
    ---
    * Compute sounds['message_frame'] using 'last_frame_time' and 
      'frames_since_cycle_start'
    * Compute sounds['speaker_frame'] by compensating for buffer
    * Fixes wraparound issues and checks that the rows are in order, or
      warns if they are not.
    * Fits 'message_frame' to 'message_time_s' to determine the link between
      jack frames and clock time.
    * Uses that fit to compute 'speaker_time_s', the estimate clock time at
      which the sound played.
    
    Arguments
    ---
    sounds : DataFrame
    octopilot_session_name : str
        Name of the session, used only for printing warning messages
    suppress_order_warnings : bool
        If True, suppress warning about the rows in `sounds` being out of
        order.
    
    Returns: DataFrame
        This is `sounds` with a few columns added, notably 'speaker_time_s'.
        The order of the rows is likely different.
    """
    ## Account for buffering delay
    # This is mostly from paclab.parse.load_sounds_played
    # Calculate message_frame, the frame number at the time the message was sent
    # This corresponds to 'sound_time', approximately, in clock time
    sounds['message_frame'] = (
        sounds['last_frame_time'] + 
        sounds['frames_since_cycle_start'])

    # Calculate the frame when the sound comes out
    # This will be rounded up to the next block, and then plus N_buffer_blocks
    sounds['speaker_frame'] = (
        (sounds['message_frame'] // 1024 + 1) * 1024
        + 2 * 1024)

    # frames_since_cycle_start is generally between 70 and 280, indicating 
    # a 0.3-1.5 ms delay within a 5.33 ms frame. 
    # But it's bi- or tri-modal, and seems potentially more delayed on 
    # continuation frames (presumably because of the cost of sending a ZMQ message)
    # I'm concerned that it's more than half a frame on 0.3% of sounds. What's
    # happening on these sounds?


    ## Find the best fit between message_time_s and message_frame
    # Calculate the (almost linear) relationship between jack frame numbers
    # and session time. This must be done separately for each pi since each 
    # has its own frame clock
    #
    # In the course of this fit, we also fix wraparound issues
    pilot2jack_frame2session_time = {}
    new_sounds_played_df_l = []

    # Iterate over rpi
    for pilot, subdf in sounds.groupby('rpi'):
        # Deal with wraparound
        # message_frame can wrap around 2**31 to -2**31
        # It no longer seems to be necessary to convert to int64
        # But can the wraparound still happen?
        #int32_info = np.iinfo(np.int32)
        subdf['message_frame'] = subdf['message_frame'].astype(np.int64)
        subdf['speaker_frame'] = subdf['speaker_frame'].astype(np.int64)
        
        # Detect by this huge offset
        # There can occasionally be small offsets (see below)
        if np.diff(subdf['message_frame']).min() < -.9 * (2**32):
            print("warning: integer wraparound detected in message_frame")
            fix_mask = subdf['message_frame'] < 0
            
            # Fix both message_frame and speaker_frame
            subdf.loc[fix_mask, 'message_frame'] += 2 ** 32
            subdf.loc[fix_mask, 'speaker_frame'] += 2 ** 32
        
        # Error check ordering
        # Not sure why this happens, but sometimes two sounds are played in
        # the same frame
        diff_time = np.diff(subdf['message_frame'])
        n_out_of_order = np.sum(diff_time < 0)
        if n_out_of_order > 0 and not suppress_order_warnings:
            print(
                f"warning: {octopilot_session_name}: {n_out_of_order} rows of sounds_played_df "
                "out of order by at worst {} frames".format(diff_time.min())
                )
        
        # Fit from message_frame to session time
        pilot2jack_frame2session_time[pilot] = scipy.stats.linregress(
            subdf['message_frame'].values,
            subdf['message_time_s'].values,
        )

        # This should be extremely good because it's fundamentally a link between
        # a time coming from datetime.datetime.now() and a time coming from 
        # jackaudio's frame measurement on the same pi
        # If not, probably an xrun or jack restart occurred
        # It appears the true sampling rate of the Hifiberry is ~192002 +/- 1
        # 1/pilot2jack_frame2session_time[pilot].slope
        if (1 - pilot2jack_frame2session_time[pilot].rvalue) > 1e-8:
            print("warning: {}: rvalue was {:.9f} on {}".format(
                octopilot_session_name,
                pilot2jack_frame2session_time[pilot].rvalue,
                pilot))

        # Store the version with times fixed for wraparound
        new_sounds_played_df_l.append(subdf)

    # Reconstruct sounds DataFrame
    # message_frame and speaker_frame are now int64 and wraparound-free
    sounds = pandas.concat(new_sounds_played_df_l).sort_values('message_time')

    # Add a column for diff between frames, useful for detecting continuations
    sounds['speaker_frame_diff'] = sounds['speaker_frame'].diff()
        
    # Use that fit to estimate when the sound played in the session timebase
    speaker_time_l = []
    for pilot, subdf in sounds.groupby('rpi'):
        # Convert speaker_time_jack to behavior_time_rpi01
        speaker_time = np.polyval([
            pilot2jack_frame2session_time[pilot].slope,
            pilot2jack_frame2session_time[pilot].intercept,
            ], subdf['speaker_frame'])
        
        # Store these
        speaker_time_l.append(pandas.Series(speaker_time, index=subdf.index))

    # Concat the results and add to sounds_played_df
    concatted = pandas.concat(speaker_time_l)
    sounds['speaker_time_s'] = concatted
    
    return sounds

def join_sound_plans_on_sounds(sounds, sound_plans):
    """Join `sound_plans` on `sounds`
    
    `sound_plans` is the list of sounds that were planned to play
    `sounds` is the sounds that were actually logged as playing
    
    This function joins the relevant columns of `sound_plans` onto `sounds`.
    
    Currently the alignment is often broken!
    
    Workflow:
    * Drop rows in `sounds` corresponding to a continuation
    * Number sounds within each trial * rpi and call this "n_sound"
    * Join on sound_plans
    * Reset index on sounds
    
    Returns: sounds, sound_plans
        sounds : DataFrame
            This will always be shorter than the original sounds, because
            continuation frames were dropped.
            The column 'diff_speaker_time_s' is recomputed after this drop.
            
            The following columns are added:
            'n_sound': a cumcount of sound within each trial * rpi
            'n_sound_plan' : matched to the column in `sound_plans`
            'side', 'gap_chunks': from `sound_plans`
            
            This DataFrame is indexed by trial_number * rpi * n_sound
            
        sound_plans : DataFrame
            This will be the same length as the original sound_plans
            The column `n_sound_plan` has been added
    """
    ## Keep a copy for debugging
    orig_sounds = sounds.copy()
    

    ## Drop continuation sounds 
    # TODO: handle the case where this drops a real sound, because of a glitch
    # or some edge case around end of trial
    sounds = sounds[sounds['speaker_frame_diff'] != 1024].drop(
        'speaker_frame_diff', axis=1)

    # Recalculate diff time after continuation sounds dropped
    # Use this shift so that diff_speaker_time_s is the gap time after each sound,
    # not before, so it matches sound_plan
    sounds['diff_speaker_time_s'] = sounds['speaker_time_s'].diff().shift(-1)

    
    ## Align sounds and sound_plans
    # sound_plans is the sounds that were planned to play
    # sounds is (ideally) the sounds that actually played
    # there can be fewer sounds than planned, if the trial ended early
    # there can be many more sounds than planned, if the trial went long, 
    # in which case the plan wraps around
    
    # First rename "identity" to "rpi" so that this key is labeled the same
    sound_plans = sound_plans.rename(columns={'identity': 'rpi'})

    # Count the length of each plan
    sound_plan_lengths = sound_plans.groupby(['trial_number', 'rpi']).size()

    # Cumcount the sounds within each trial * rpi
    sounds['n_sound'] = sounds.groupby(
        ['trial_number', 'rpi']).cumcount()
    sound_plans['n_sound_plan'] = sound_plans.groupby(
        ['trial_number', 'rpi']).cumcount()
    
    # Compute "n_sound_plan" for the sounds, which is "n_sound" mod "len_plan"
    sounds = sounds.join(
        sound_plan_lengths.rename('len_plan'), on=['trial_number', 'rpi'],
        validate='m:1')
    sounds['n_sound_plan'] = sounds['n_sound'].mod(sounds['len_plan'])

    # Join the plan on the sound, ensuring it is a many-to-one join, and
    # that we retain the same number of sounds
    sounds = pandas.merge(sounds, sound_plans,
        left_on=['trial_number', 'rpi', 'n_sound_plan'],
        right_on=['trial_number', 'rpi', 'n_sound_plan'],
        how='left',
        validate='m:1',
        )

    # The only allowable null value is the last entry in diff_speaker_time_s
    assert not sounds.drop('diff_speaker_time_s', axis=1).isnull().any().any()
    assert not sounds['diff_speaker_time_s'].iloc[:-1].isnull().any()

    # Drop 'gap' and 'time', because only 'gap_chunks' (a quantized approximation
    # of 'gap') matters, because that is what was used to generate the sound
    sounds = sounds.drop(['gap', 'time', 'len_plan'], axis=1)


    ## Error check that gap_chunks matches diff_speaker_time_s
    # TODO: this is quite often wrong, need to fix
    
    # Reindex
    sounds = sounds.set_index(['trial_number', 'rpi', 'n_sound'])
    sound_plans = sound_plans.set_index(['trial_number', 'rpi', 'n_sound_plan'])
    
    # For this error check, exclude the last sound of each trial, because on
    # those sounds the next sound never occurred, and an ITI happened instead
    index_of_last_sound_per_trial = sounds.groupby(
        ['trial_number', 'rpi']).apply(
        lambda df: df.iloc[-2:].droplevel(['trial_number', 'rpi'])
        ).index

    # Drop the last sound of each trial
    to_check = sounds.drop(index_of_last_sound_per_trial)
    
    # Also exclude any trial where the gap_chunks was 1, because I think
    # this doesn't work properly
    # TODO: fix this
    min_gap_chunk = sound_plans.groupby('trial_number')['gap_chunks'].min()
    drop_trials = min_gap_chunk.index[min_gap_chunk.values == 1]

    # Drop the trials with min_gap_chunk == 1
    # ignore errors because it may already have been dropped above
    to_check = to_check.drop(drop_trials, errors='ignore')
    
    # Compute the estimate time between sounds
    # This assumes that the duration of the sound is 2 chunks!
    to_check['est_diff'] = (to_check['gap_chunks'] + 2) * 1024 / 192000
    
    # Compute the error in the estimate
    # typical median abs(error) is ~2 us
    to_check['err'] = to_check['diff_speaker_time_s'] - to_check['est_diff']

    # Compute fraction of sounds that are extremely off (0.1 ms)
    # This appears to be pretty accurate in most cases, but sometimes off for
    # the last sound in each trial (even after dropping the actual last above)
    bad_trials = (to_check['err'].abs() > 1e-4).groupby('trial_number').any()
    
    # This is off so often that it's not worth warning about
    #~ # Warn
    #~ if bad_trials.sum() > 1:
        #~ print(
            #~ f'warning: sound_plans does not align well with sounds on '
            #~ f'{bad_trials.sum()} / {len(bad_trials)} trials')
    
    return sounds, sound_plans
    