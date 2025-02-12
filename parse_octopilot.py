"""For parsing octopilot files

"""

def str2dt(s):
    """Transform string `s` into a datetime in timezone `America/NewYork`"""
    tz = pytz.timezone('America/New_York')
    return datetime.datetime.fromisoformat(s).astimezone(tz)

def load_session(octopilot_root, octopilot_session_name):
    """Load data from octopilot session
    
    octopilot_root : str
        Path to root of octopilot data
    
    octopilot_session_name : str
    """
    ## Form session dir
    octopilot_year, octopilot_month = octopilot_session_name.split('-')[:2]
    octopilot_session_dir = os.path.join(
        octopilot_root, octopilot_year, octopilot_month, octopilot_session_name)


    ## Load behavior data
    sounds = pandas.read_table(
        os.path.join(octopilot_session_dir, 'sounds.csv'), sep=',')
    sound_plans = pandas.read_table(
        os.path.join(octopilot_session_dir, 'sound_plans.csv'), sep=',')
    trials = pandas.read_table(
        os.path.join(octopilot_session_dir, 'trials.csv'), sep=',')
    pokes = pandas.read_table(
        os.path.join(octopilot_session_dir, 'pokes.csv'), sep=',')
    flashes = pandas.read_table(
        os.path.join(octopilot_session_dir, 'flashes.csv'), sep=',', header=None)

    # Rename this one to indicate that it is the time the message was
    # sent about the sound, which is definitely not the time it played
    # TODO: do this in octopilot itself
    sounds = sounds.rename(columns={'sound_time': 'message_time'})

    # TODO: fix this missing header
    flashes.columns = ['trial_number', 'rpi', 'flash_time']

    # Coerce
    flashes['flash_time'] = flashes['flash_time'].apply(str2dt)
    sounds['message_time'] = sounds['message_time'].apply(str2dt)
    pokes['poke_time'] = pokes['poke_time'].apply(str2dt)
    trials['start_time'] = trials['start_time'].apply(str2dt)
    trials['reward_time'] = trials['reward_time'].apply(str2dt)


    ## Define session_start_time
    # Take the time of the first trial on the desktop
    # It doesn't really matter whether this is the same time on all of the
    # other pis. This is just a point of reference to convert datetimes to floats
    session_start_time = trials['start_time'].iloc[0]

    # Convert datetimes to time in seconds since session_start_time
    sounds['message_time_s'] = (
        sounds['message_time'] - session_start_time).dt.total_seconds()


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
        diff_time = np.diff(subdf['message_frame'])
        n_out_of_order = np.sum(diff_time < 0)
        if n_out_of_order > 0:
            print(
                "warning: {} rows of sounds_played_df ".format(n_out_of_order) +
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
        if (1 - pilot2jack_frame2session_time[pilot].rvalue) > 1e-9:
            print("warning: rvalue was {:.3f} on {}".format(
                pilot2jack_frame2session_time[pilot].rvalue,
                pilot))
            print("speaker_time_in_session will be inaccurate")

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


    ## Join the sound_plans on the sounds
    # This actually works now that we have an ITI

    # Drop continuation sounds 
    # TODO: handle the case where this drops a real sound, because of a glitch
    # or some edge case around end of trial
    sounds = sounds[sounds['speaker_frame_diff'] != 1024].drop('speaker_frame_diff', axis=1)

    # Recalculate diff time after continuation sounds dropped
    # Use this shift so that diff_speaker_time_s is the gap time after each sound,
    # not before, so it matches sound_plan
    sounds['diff_speaker_time_s'] = sounds['speaker_time_s'].diff().shift(-1)

    # Rename identity to rpi
    sound_plans = sound_plans.rename(columns={'identity': 'rpi'})

    # Number sounds withing trial * rpi for each
    sounds['n_sound'] = sounds.groupby(['trial_number', 'rpi']).cumcount()
    sound_plans['n_sound'] = sound_plans.groupby(['trial_number', 'rpi']).cumcount()

    # Index
    sounds = sounds.set_index(['trial_number', 'rpi', 'n_sound'])
    sound_plans = sound_plans.set_index(['trial_number', 'rpi', 'n_sound'])

    # Join
    sounds = sounds.join(sound_plans, rsuffix='_planned')


    ## Error check
    # diff_speaker_time_s will never match gap_time on the last sound of a trial,
    # because on those sounds the next sound never occurred, and an ITI happened
    # instead
    index_of_last_sound_per_trial = sounds.groupby(
        ['trial_number', 'rpi']).apply(
        lambda df: df.iloc[-1:].droplevel(['trial_number', 'rpi'])
        ).index

    # Drop 'gap' and 'time', because only 'gap_chunks' (a quantized approximation
    # of 'gap') matters, because that is what was used to generate the sound
    sounds = sounds.drop(['gap', 'time'], axis=1)

    # Drop the last sound of each trial
    to_check = sounds.drop(index_of_last_sound_per_trial)

    # This should look like a perfectly straight line (to nanosecond)
    # This just checks that sounds lines up with sounds_plan, nothing about
    # actual synchronization electrically
    # plot(to_check['gap_chunks'], to_check['diff_speaker_time_s'], '.')

    # Reset index for backwards compat
    sounds = sounds.reset_index()


    ## Put rpi on columns of flashes
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


    ## Return
    return {
        'sounds': sounds,
        'sound_plans': sound_plans,
        'trials': trials,
        'pokes': pokes,
        'flashes': flashes,
        'flash_wrt_session_start': flash_wrt_session_start,
        }
