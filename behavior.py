"""Functions for dealing with behavior videos"""

import datetime
import os
import glob
import scipy.optimize
import pandas
import my

def get_video_timing_metadata(video_filenames, verbose=False):
    """Get timing metadata for every video
    
    Arguments
    ---
    video_filenames : list-like
        A list of full video filenames
    verbose : boolean
        If True, prints warning on skipped files

    Each filename is processed in turn. Filenames that do not end with *.avi
    are skipped. Filenames that don't appear to contain an entire timestamp
    are skipped. The filename is parsed into a time which is taken as the
    start time fo the video. The modification time of the file is taken.
    
    Returns: DataFrame, with columns
        filename : video filename
        start : the start time obtained from parsing the filename
        mod : the modification time of the file itself
    """
    # Format them
    res_l = []
    for video_filename in video_filenames:
        # Skip if not avi
        if not video_filename.endswith('.avi'):
            if verbose:
                print("doesn't end with avi; skipping " + video_filename)
            continue
        
        # Shorten
        short_video_filename = os.path.split(video_filename)[1]
        
        # Split on hyphens
        split = short_video_filename.replace('.avi', '').split('-')
        
        # The last token should be 6 digits (microseconds)
        # But sometimes it skips the microseconds, I think if it got screwed up
        if len(split[-1]) != 6:
            if verbose:
                print("doesn't end with full timestamp; skipping " + 
                    video_filename)
            continue
        
        # The timestamp should be the next to last one
        timestamp = split[-2]
        assert timestamp[8] == 'T'
        full_timestamp = timestamp + '.' + split[-1]
        
        # Format
        dt = datetime.datetime.strptime(full_timestamp, "%Y%m%dT%H%M%S.%f")
        
        # Mod time
        # This is approximately the end time
        mod_ts = my.misc.get_file_time(video_filename, human=False)
        mod_time = datetime.datetime.fromtimestamp(mod_ts)
        
        # Store
        res_l.append((short_video_filename, dt, mod_time))

    # DataFrame it
    df = pandas.DataFrame.from_records(
        res_l, columns=['filename', 'start', 'mod'])
    df['approx_duration_video'] = df['mod'] - df['start']
    
    return df

def _match_videos_with_behavior(video_time, behavior_time, threshold=10):
    """Align video times and behavior times
    
    Parameters
    ---
    video_time : pandas.Series
        index : video filename
        values : time of video
    behavior_time : pandas.Series
        index : behavioral session name
        values : time of session
    threshold : numeric
        No assignment can be off by more than this many seconds
    
    First only behavior times that are within `threshold` seconds of
    any video time are included. Then scipy.optimize.linear_sum_assignment
    is used to find the best matching of video_time and behavior_time.
    Finally asignments that are more than `threshold` seconds apart are dropped.
    
    Returns : DataFrame, with columns
        video_filename : filename of vidoe, taken from video_time.index
        session_name : behavioral session name, taken from behavior_time.index
        cost : the cost of the assignment (difference between the times)    
    """
    # Sometimes there are behavior sessions with no matched video
    # That can really screw up the hungarian because they'll be aligned to
    # something really far away
    # So pre-process by removing any behavior sessions that don't have 
    # any nearby videos
    include_mask = my.misc.times_near_times(
        video_time.apply(lambda dt: dt.timestamp()), 
        behavior_time.apply(lambda dt: dt.timestamp()), 
        dstart=-threshold, dstop=threshold)
    behavior_time = behavior_time.loc[include_mask]

    # Align
    alignment_df = pandas.concat([video_time] * len(behavior_time), 
        axis=1)
    alignment_df.columns = behavior_time.index
    alignment_df.index = video_time.index
    alignment_df = alignment_df.sub(behavior_time)
    alignment_df = alignment_df.apply(lambda xxx: xxx.dt.total_seconds())
    alignment_df = alignment_df.abs()

    # Hungarian
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(alignment_df)
    aligned_l = []
    for rr, cc in zip(row_ind, col_ind):
        cost = alignment_df.iloc[rr, cc]
        video_filename = alignment_df.index[rr]
        session_name = alignment_df.columns[cc]
        aligned_l.append((video_filename, session_name, cost))
    aligned_df = pandas.DataFrame.from_records(
        aligned_l, columns=['video_filename', 'session_name', 'cost'])

    # Anything with a cost more than 3 is missing
    aligned_df = aligned_df[aligned_df['cost'] < threshold]
    return aligned_df

def match_videos_with_behavior(video_dir, session_df, quiet=False):
    """Match videos with behavior
    
    Loads all video filenames in `video_dir`. Extracts "sandbox_creation_time"
    from `session_df`. Uses the lower-level _match_videos_with_behavior
    to actually match these up. Extracts the inferred camera name from the
    video file and ensures this matches what is in `session_df`. Prints
    a warning if it doesn't match, and also if there are behavior sessions
    without a matched video.
    
    
    Parameters
    ---
    video_dir : string
        Directory that will be searched for videos
    
    session_df : pandas.DataFrame
        This should come from paclab.parse.parse_sandboxes
        
    quiet : bool
        If True, print warnings to stdout
        These are in any case returned in `output_txt`
    
    
    Returns: aligned_videos_df, output_txt
        aligned_videos_df : DataFrame
            index : MultiIndex data * mouse
            columns : 'session_name', 'video_filename', 'approx_duration', 
                'approx_duration_hdf5', 'approx_duration_video', 'camera_name'
        
        output_txt : string
            May contain printed warnings
    """
    ## This is where warnings will go
    output_txt = ''
    
    
    ## Get a list of all videos
    video_filenames = glob.glob(os.path.join(video_dir, '*.avi'))


    ## Load video timing metadata
    df = get_video_timing_metadata(video_filenames)
    video_time = df.set_index('filename')['start']


    ## Align sessions and videos
    # Drop any missing sandbox_creation_time or this won't work
    session_df = session_df[~session_df['sandbox_creation_time'].isnull()]
    
    # Use sandbox_creation_time to avoid the latency of the person entering 
    # the mouse weight
    behavior_time = session_df[
        'sandbox_creation_time'].droplevel('mouse').apply(
        lambda s: datetime.datetime.fromisoformat(s))

    # Align
    aligned_df = _match_videos_with_behavior(
        video_time, behavior_time, threshold=10)


    ## Form aligned_videos_df
    # Store in session_df
    session_df = session_df.join(
        aligned_df.set_index('session_name')['video_filename'])

    # Also join video duration
    session_df = session_df.join(
        df.set_index('filename')['approx_duration_video'], on='video_filename')

    # Include only columns relevant to video analysis (e.g., no stim params)
    aligned_videos_df = session_df.reset_index()[[
        'date', 'mouse', 'session_name', 'video_filename', 'approx_duration', 
        'approx_duration_hdf5', 'approx_duration_video', 'camera_name']
        ].set_index(['date', 'mouse']).sort_index()

    
    ## If there are duplicate sessions by day, the rest won't work
    if aligned_videos_df.index.duplicated().any():
        output_txt += (
            "error: cannot check for missing video if there are "
            "duplicated sessions\n")
    else:    
        ## Extract inferred camera name
        # Extract the inferred camera name from the filename
        aligned_videos_df['inferred_camera_name'] = (
            aligned_videos_df['video_filename'].apply(
            lambda s: s[:7] if not pandas.isnull(s) else ''))

        # Ensure these match
        mistaken_cameras_mask = (
            aligned_videos_df['camera_name'] != 
            aligned_videos_df['inferred_camera_name'])
        mistaken_cameras_sessions = aligned_videos_df.loc[
            mistaken_cameras_mask.values]
        mistaken_cameras_sessions = mistaken_cameras_sessions[
            ~mistaken_cameras_sessions['video_filename'].isnull()]
        if len(mistaken_cameras_sessions) > 0:
            output_txt += ("warning: some cameras were mis-inferred:\n")
            output_txt += ("{}\n\n".format(
                str(mistaken_cameras_sessions[['video_filename', 'camera_name']])
                ))


        ## Identify sessions with missing video
        # Identify sessions where the video is missing
        # Ignore ones from today because they haven't been uploaded yet
        to_check = aligned_videos_df.drop(datetime.date.today(), errors='ignore')
        sessions_missing_video = to_check.loc[
            aligned_videos_df['video_filename'].isnull()]
        if len(sessions_missing_video) > 0:
            output_txt += ("warning: {}/{} sessions are missing video\n".format(
                len(sessions_missing_video),
                len(aligned_videos_df),
                ))
            output_txt += str(
                sessions_missing_video[['session_name', 'camera_name']])
            output_txt += "\n\n"
    
    
    ## Optionally output warnings
    if not quiet:
        print(output_txt)
    
    return aligned_videos_df, output_txt
    