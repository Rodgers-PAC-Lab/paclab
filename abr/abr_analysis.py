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