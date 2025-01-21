"""Functions for signal processing of ABR data

"""
import scipy.signal
import numpy as np

def identify_click_times(speaker_signal_V, threshold_V,
    highpass_freq=50, sampling_rate=16000, refrac_samples=750):
    """Identify when the clicks occurred
    
    speaker_signal : array
        Voltage on the speaker, in V
    
    threshold_V : scalar
        Minimum value of a click, in V
    
    refrac_samples : scalar
        Minimum time between clicks ("refractory period") in samples
    
    Returns: speaker_signal_hp, peak_times
        speaker_signal_hp : array
            The speaker siganl highpassed
        peak_times : array
            Peak times in samples
    """
    ## Highpass audio data above 50 Hz to remove noise
    # This remove some baseline noise, but also makes the big clicks "ring"
    nyquist_freq = sampling_rate / 2.0
    ad_ahi, ad_bhi = scipy.signal.butter(
        2, (highpass_freq / nyquist_freq), btype='highpass')
    speaker_signal_hp = scipy.signal.lfilter(ad_ahi, ad_bhi, speaker_signal_V)

    # Find peaks
    pos_peaks, pos_props = scipy.signal.find_peaks(
        speaker_signal_hp, height=threshold_V, distance=refrac_samples)

    neg_peaks, neg_props = scipy.signal.find_peaks(
        -speaker_signal_hp, height=threshold_V, distance=refrac_samples)

    # Concatenate pos and neg peaks,
    #  then drop ringing/overshoot peaks during refractory period
    peak_times = np.sort(np.concatenate([pos_peaks, neg_peaks]))

    return speaker_signal_hp, peak_times
