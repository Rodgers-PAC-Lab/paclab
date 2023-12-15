"""Functions for loading and synchronizing behavioral, neural, video data"""

import os
import struct
import matplotlib.pyplot as plt
import numpy as np
import my.neural
import my.misc
import paclab
import datetime
import Adapters
import MCwatch.behavior
import pandas
import scipy.stats
import ns5_process.AudioTools
import tables
import pytz


def load_analog_data(analog_packed_filename):
    """Check filesize on analog_packed_filename and return memmap"""
    # Check size
    analog_packed_file_size_bytes = os.path.getsize(analog_packed_filename)
    analog_packed_file_size_samples = analog_packed_file_size_bytes // 32 // 2
    assert analog_packed_file_size_samples * 32 * 2 == analog_packed_file_size_bytes

    # Memmap the analog data
    analog_mm = np.memmap(
        analog_packed_filename, 
        np.int16, 
        'r', 
        offset=0,
        shape=(analog_packed_file_size_samples, 32)
    )    
    
    return analog_mm

def load_neural_data(neural_packed_filename):
    packed_file_size_bytes = os.path.getsize(neural_packed_filename)
    packed_file_size_samples = (packed_file_size_bytes - 8) // 64 // 2
    assert packed_file_size_samples * 64 * 2 + 8 == packed_file_size_bytes
    neural_mm = np.memmap(
        neural_packed_filename, 
        np.int16, 
        'r', 
        offset=8,
        shape=(packed_file_size_samples, 64)
    )   
    
    return neural_mm

def get_recording_start_time(trig_signal):
    """Return the time (in samples) of the recording start pulse"""
    # Find threshold crossings
    # 10.0V = 32768 (I think?), so 3.3V = 10813
    # Take the first sample that exceeds roughly half that
    # We expect trig signal to last 100 ms (I think?), which is 2500 samples
    # There is a pulse about 6000 samples long at the very beginning, which I
    # think is when the nosepoke is initialized
    trig_time_a, trig_duration_a = (
        MCwatch.behavior.syncing.extract_onsets_and_durations(
        trig_signal, delta=5000, verbose=False, maximum_duration=5000))
    assert len(trig_time_a) == 1
    assert trig_duration_a[0] > 2495 and trig_duration_a[0] < 2540
    trig_time = trig_time_a[0]
    
    return trig_time

