import math
import os.path
import struct
import numpy as np
import scipy
import pickle
import pandas
import socket

def laterality_check(channel, speaker_side):
    """
    Determine if the sound is playing on the contralateral or ipsilateral side from the recording on a particular channel.
    Only works with channels LV and RV.
    LR doesn't really have an ipsilateral or contralateral side since it records from both ears.

    Parameters:
        channel: string 'LV' or 'RV', which channel the data is from
        speaker_side: string 'L' or 'R', which side the speaker plays on

    Returns:
        laterality: string 'ipsilateral' or 'contralateral'. Returns np.nan if it gets invalid input.
    """
    if speaker_side == 'L':
        if channel == 'LV':
            laterality = 'ipsilateral'
        elif channel == 'RV':
            laterality = 'contralateral'
        else:
            print("Invalid channel and speaker side config: ",channel,speaker_side)
            laterality = np.nan
    elif speaker_side == 'R':
        if channel == 'RV':
            laterality = 'ipsilateral'
        elif channel == 'LV':
            laterality = 'contralateral'
        else:
            print("Invalid channel and speaker side config: ", channel, speaker_side)
            laterality = np.nan
    else:
        print("Invalid channel and speaker side config: ", channel, speaker_side)
        laterality = np.nan
    return laterality

def pre_or_post(timepoint):
    """
    Determine if the timepoint is pre or post hearing loss.
    Timepoints are usually labeled like apreA, postB, etc.

    Parameters:
        timepoint: string of timepoint to check

    Returns:
        res: string 'pre' or 'post'. Returns np.nan if timepoint doesn't contain 'pre' or 'post'
    """
    if 'pre' in timepoint:
        res = 'pre'
    elif 'post' in timepoint:
        res = 'post'
    else:
        print('ERROR, ' + str(timepoint) + 'not pre or post!')
        res = np.nan
    return res

def join_cohort_info_to_df(df, cohort_experiments, join_on=['date','mouse'],
                           join_HL=True, join_timepoint=True, join_sex=False,
                           join_strain=False, join_genotype=False, join_age=False):
    """
    Takes any df that includes mouse and date, then adds useful info from cohort_experiments

    Arguments:
        df: pandas Dataframe with mouse and date somewhere in the index or on the columns.
        cohort_experiments: pandas Dataframe that's based on mouse_info.csv
        join_on: list of str, the columns that are the same in both dataframes which you join on.
            Pretty much always should be 'date' and 'mouse'
        join_HL: boolean, do you want to add hearing loss status to df
        join_timepoint: boolean, do you want to add testing timepoint (pre-HL, post-HL, etc) to df
        join_sex: boolean, do you want to add mouse sex to df
        join_strain: boolean, do you want to add mouse strain to df
        join_genotype: boolean, do you want to add mouse genotype to df
        join_age: boolean, do you want to add mouse age to df
    Returns:
        df: pandas Dataframe
            The original dataframe with your chosen columns added. Keeps its original index.
    """
    if 'timepoint' in join_on and join_timepoint==True:
        print("You're trying to join on timepoint but also append timepoint, which won't work.")
        print("Changing join_timepoint==False so the join works. Timepoint will be in index not columns.")
        join_timepoint=False
    columns_ser = pandas.Series({
        'HL':   join_HL,
        'timepoint': join_timepoint,
        'sex': join_sex,
        'strain': join_strain,
        'genotype': join_genotype,
        'age': join_age
    })

    # Use booleans to get a list of the join_ columns you set True
    columns_to_join = columns_ser.loc[columns_ser==True].index
    # Check to see if you're trying to join a column that already exists in df
    df_idx_l = df.index.names
    for col in columns_to_join:
        if col in df_idx_l:
            columns_to_join = columns_to_join.drop(col)
            print('Tried to join  the column ' + col + ', but it already exists in the original df')


    # Save the old index to re-apply later
    df_idx = df.index.to_frame()
    # Change index of df and cohort_experiments to match
    df = df.reset_index().set_index(join_on)
    cohort_experiments = cohort_experiments.reset_index().set_index(join_on)

    # Join selected columns from cohort_experiments to df
    df = df.join(cohort_experiments[columns_to_join], on=join_on)
    # Set the df index back how it was
    df.index = pandas.MultiIndex.from_frame(df_idx)
    return df