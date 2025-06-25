"""Objects to run the ABR GUI

TODO:
* Impedance measurement
* Plot lines at max so we can see railing, or flag when it's totally out of range
* Allow flipping live switch
* invert color order in ABR plot
* make a mean ABR

Timing parameters:
* OscilloscopeWidget.update_interval_ms
    milliseconds between update calls to all plotting objects
    This should be sufficiently longer than the actual time taken
* MainWindow.update_interval_ms
    milliseconds between update calls to main window (just text boxes)
* OscilloscopeWidget.duration_data_to_analyze_s
    Seconds of data to load from the deq
    As this gets longer, we have more information to compute the ABR, but
    it will take longer to analyze.

Labels:
* The text label "data analyzed (s)" indicates how much is being used by
the GUI, which will be at most duration_data_to_analyze_s.
* The text label "packets in memory" is the length of the deque available to 
the GUI. 
* The text label "data collected" indicates how much data has been read
over the entire experiment, even that which is no longer in memory.
* time taken and debug timing interval??
"""

import datetime
import scipy.signal
import paclab.abr
import numpy as np
import pandas
import matplotlib.mlab
import PyQt5.QtWidgets
import PyQt5.QtCore
import pyqtgraph as pg

# Temporary workaround
# In the main branch, this was just paclab.abr
# Now everything's been moved into a subfolder
# In any case we need to remove all references to paclab.abr.abr and 
# paclab.abr.abr_gui
import paclab.abr.abr

# TODO: move this to a shared location
def psd(data, NFFT=None, Fs=None, detrend='mean', window=None, noverlap=None, 
    scale_by_freq=None, **kwargs):
    """Compute power spectral density.
    
    A wrapper around mlab.psd with more documentation and slightly different
    defaults.
    
    Arguments
    ---
    data : The signal to analyze. Must be 1d
    NFFT : defaults to 256 in mlab.psd
    Fs : defaults to 2 in mlab.psd
    detrend : default is 'mean', overriding default in mlab.psd
    window : defaults to Hanning in mlab.psd
    noverlap : defaults to 0 in mlab.psd
        50% or 75% of NFFT is a good choice in data-limited situations
    scale_by_freq : defaults to True in mlab.psd
    **kwargs : passed to mlab.psd
    
    Notes on scale_by_freq
    ---
    Using scale_by_freq = False makes the sum of the PSD independent of NFFT
    Using scale_by_freq = True makes the values of the PSD comparable for
    different NFFT
    In both cases, the result is independent of the length of the data
    With scale_by_freq = False, ppxx.sum() is roughly comparable to 
      the mean of the data squared (but about half as much, for some reason)
    With scale_by_freq = True, the returned results are smaller by a factor
      roughly equal to sample_rate, but not exactly, because the window 
      correction is done differently
    
    With scale_by_freq = True
      The sum of the PSD is proportional to NFFT/sample_rate
      Multiplying the PSD by sample_rate/NFFT and then summing it
        gives something that is roughly equal to np.mean(signal ** 2)
      To sum up over a frequency range, could ignore NFFT and multiply
        by something like bandwidth/sample_rate, but I am not sure.
    With scale_by_freq = False
      The sum of the PSD is independent of NFFT and sample_rate
      The sum of the PSD is slightly more than np.mean(signal ** 2)
      To sum up over a frequency range, need to account for the number of
        points in that range, which depends on NFFT.
    In both cases
      The sum of the PSD is independent of the length of the signal
    The reason that the answers are not proportional to each other
    is because the window correction is done differently. 
    
    scale_by_freq = True generally seems to be more accurate
    I imagine scale_by_freq = False might be better for quickly reading
    off a value of a peak    
    """
    # Run PSD
    Pxx, freqs = matplotlib.mlab.psd(
        data,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        scale_by_freq=scale_by_freq,
        **kwargs,
        )

    # Return
    return Pxx, freqs

class OscilloscopeWidget(PyQt5.QtWidgets.QWidget):
    def __init__(self, 
        abr_device, 
        update_interval_ms=500,
        duration_data_to_analyze_s=60, 
        neural_scope_xrange_s=5,
        neural_scope_yrange_mV=200, # max achievable is 187.500 at gain=24
        highpass_neural_scope_yrange_uV=30, # should have stdev ~1 uV
        audio_scope_xrange_s=5,
        audio_scope_yrange_mV=300,
        abr_audio_monitor_yrange_uV=5, # abslog scale
        abr_neural_yrange_uV=5,
        audio_extract_win_samples=10,
        verbose=False,
        *args, **kwargs):
        
        ## Superclass PyQt5.QtWidgets.QWidget init
        super().__init__(*args, **kwargs)
        
        
        ## Instance variables
        # abr_device, where the data comes from
        self.abr_device = abr_device
        self.verbose = verbose
        
        # Timers for continuous updating
        # Create a PyQt5.QtCore.QTimer object to continuously update the plot         
        self.timer_update_plot = PyQt5.QtCore.QTimer(self) 
        self.timer_update_plot.timeout.connect(self.update)  
        self.update_interval_ms = update_interval_ms

        # Parameters that cannot be set by user interaction
        self.abr_highpass_freq = 300
        self.abr_lowpass_freq = 3000
        self.heartbeat_highpass_freq = 20
        self.heartbeat_lowpass_freq = 1000
        self.ekg_recent_duration_window_s = 30
        self.duration_data_to_analyze_s = duration_data_to_analyze_s
        self.audio_extract_win_samples = audio_extract_win_samples

        # TODO: get this from config
        self.neural_channels_to_plot = [0, 2, 4]
        self.speaker_channel = 7

        # Parameters that can be set by user interaction
        self.neural_scope_xrange_s = neural_scope_xrange_s
        self.neural_scope_yrange_mV = neural_scope_yrange_mV
        self.highpass_neural_scope_yrange_uV = highpass_neural_scope_yrange_uV
        self.audio_scope_xrange_s = audio_scope_xrange_s
        self.audio_scope_yrange_mV = audio_scope_yrange_mV
        self.abr_audio_monitor_yrange_uV = abr_audio_monitor_yrange_uV
        self.abr_neural_yrange_uV = abr_neural_yrange_uV

        # Parameters that we can't set until we have data
        self.heart_rate = -1
        
        # Debug
        self.print_timing_information = False
        
        
        ## ABR extraction parameters
        # TODO: Link these to the audio generation code in some way
        # Set up monitor
        # Categorize the clicks (these are now in uV instead of mV)

        # These are the values from main4.py, which CR hand-chose to match
        # the values that I think we standardly used for most of 2024
        autopilot_voltages = 10 ** np.array([
            [-3.2  , -3.2],
            [-2.7  , -2.7],
            [-2.1  , -2.1],
            [-1.9  , -1.9],
            [-1.6  , -1.6],
            [-1.1  , -1.1],
            [-0.6  , -0.6],
            ])
        
        # These are the values from 250109_new_amplitudes
        # The first column is amplitude specified in autopilot
        # The second column is the observed voltage in V
        # Each level differs by 0.2 log units (4 dB)
        autopilot_voltages = np.array([
            [0.01, 0.19],
            [6.3e-3, 0.12],
            [4.0e-3,0.075],
            [2.5e-3, 0.047],
            [1.6e-3, 0.03],
            [1.0e-3, 0.018],
            [6.3e-4, 0.012],
            [4.0e-4, 0.0078],
            [2.5e-4, 0.004],
            [1.6e-4, 0.0028],
            [1.0e-4, 0.0018],
            [6.3e-5, 0.0012],
            [4.0e-5, 0.00075],
            ])
        
        # bins should be increasing, not decreasing
        autopilot_voltages = autopilot_voltages[::-1]
        
        # all amplitudes are in uV now
        autopilot_voltages[:, 1] = autopilot_voltages[:, 1] * 1e6
        
        # Define the amplitude cuts to be between the nominal voltages
        log10_voltage = np.log10(autopilot_voltages[:, 1])
        amplitude_cuts = (log10_voltage[1:] + log10_voltage[:-1]) / 2
        
        # Add a first and last amplitude cut
        diff_cut = np.mean(np.diff(amplitude_cuts))
        amplitude_cuts = np.concatenate([
            [amplitude_cuts[0] - diff_cut],
            amplitude_cuts,
            [amplitude_cuts[-1] + diff_cut],
            ])
        
        # Store the cuts and the labels
        # TODO: label in dB SPL, or at least dB
        self.amplitude_cuts = amplitude_cuts
        self.amplitude_labels = autopilot_voltages[:, 0]
        
        
        ## Initialize the plot_widget which actually does the plotting
        # Initializing the pyqtgraph widgets
        self.neural_plot_widget = pg.PlotWidget() 
        self.highpass_neural_plot_widget = pg.PlotWidget() 
        self.speaker_plot_widget = pg.PlotWidget()
        self.psd_monitor_widget = pg.PlotWidget()
        self.abr_pos_audio_monitor_widget = pg.PlotWidget()
        self.abr_neg_audio_monitor_widget = pg.PlotWidget()
        self.abr_neural_ch0_monitor_widget = pg.PlotWidget()
        self.abr_neural_ch2_monitor_widget = pg.PlotWidget()
        self.abr_neural_ch4_monitor_widget = pg.PlotWidget()
        
        # Size them
        for widget in [
            self.neural_plot_widget,
            self.highpass_neural_plot_widget,
            self.speaker_plot_widget,
            ]:
            
            widget.setFixedHeight(150)
            widget.setFixedWidth(800)

        for widget in [
            self.psd_monitor_widget,
            self.abr_pos_audio_monitor_widget,
            self.abr_neg_audio_monitor_widget,
            self.abr_neural_ch0_monitor_widget,
            self.abr_neural_ch2_monitor_widget,
            self.abr_neural_ch4_monitor_widget,
            ]:
            
            widget.setFixedHeight(150)
            widget.setFixedWidth(262)
        
        
        ## Debugging
        self.times_l = []
        
        ## Set the layout
        # Primarily it will be vertical, with each element being horizontal
        self.layout = PyQt5.QtWidgets.QVBoxLayout(self) 
        
        
        ## Top row: a row of neural widgets
        # Horizontal: plot widget, and then Grid of params
        row_neural_plot = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(row_neural_plot)
        row_neural_boxes = PyQt5.QtWidgets.QGridLayout()
        row_neural_plot.addWidget(self.neural_plot_widget)
        row_neural_plot.addLayout(row_neural_boxes)
        
        # Fix the width of the grid of params
        row_neural_boxes.setColumnMinimumWidth(0, 100)
        row_neural_boxes.setColumnMinimumWidth(1, 100)
        
        # Param: neural_scope_xrange_s
        self.line_edit_neural_scope_xrange_s = PyQt5.QtWidgets.QLineEdit(
            str(self.neural_scope_xrange_s))
        self.line_edit_neural_scope_xrange_s.returnPressed.connect(
            self.line_edit_neural_scope_xrange_s_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('neural scope xrange (s)'), 1, 0)
        row_neural_boxes.addWidget(self.line_edit_neural_scope_xrange_s, 1, 1)
        
        # Param: neural_scope_yrange_mV
        self.line_edit_neural_scope_yrange_mV = PyQt5.QtWidgets.QLineEdit(    
            str(self.neural_scope_yrange_mV))
        self.line_edit_neural_scope_yrange_mV.returnPressed.connect(
            self.line_edit_neural_scope_yrange_mV_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('neural scope yrange (mV)'), 0, 0)
        row_neural_boxes.addWidget(self.line_edit_neural_scope_yrange_mV, 0, 1)

        # Param: checkbox for plot_ch0
        self.checkbox_plot_ch0_neural = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch0_neural.setChecked(True)
        self.checkbox_plot_ch0_neural.stateChanged.connect(
            self.checkbox_plot_ch0_neural_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 0?'), 2, 0)
        row_neural_boxes.addWidget(self.checkbox_plot_ch0_neural, 2, 1)
        
        # Param: checkbox for plot_ch2
        self.checkbox_plot_ch2_neural = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch2_neural.setChecked(True)
        self.checkbox_plot_ch2_neural.stateChanged.connect(
            self.checkbox_plot_ch2_neural_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 2?'), 3, 0)
        row_neural_boxes.addWidget(self.checkbox_plot_ch2_neural, 3, 1)

        # Param: checkbox for plot_ch4
        self.checkbox_plot_ch4_neural = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch4_neural.setChecked(True)
        self.checkbox_plot_ch4_neural.stateChanged.connect(
            self.checkbox_plot_ch4_neural_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 4?'), 4, 0)
        row_neural_boxes.addWidget(self.checkbox_plot_ch4_neural, 4, 1)

        # Param: label for amount of data received
        self.label_analyze_data_duration_s = PyQt5.QtWidgets.QLabel()
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('data analyzed (s)'), 5, 0)
        row_neural_boxes.addWidget(self.label_analyze_data_duration_s, 5, 1)

        # Label: rms on each channel
        self.label_raw_neural_rms = PyQt5.QtWidgets.QLabel('')
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('raw rms (mV): '), 6, 0)
        row_neural_boxes.addWidget(self.label_raw_neural_rms, 6, 1)


        ## Second row: a row of highpass neural widgets
        # Horizontal: plot widget, and then Grid of params
        row_highpass_neural_plot = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(row_highpass_neural_plot)
        row_highpass_neural_boxes = PyQt5.QtWidgets.QGridLayout()
        row_highpass_neural_plot.addWidget(self.highpass_neural_plot_widget)
        row_highpass_neural_plot.addLayout(row_highpass_neural_boxes)

        # Fix the width of the grid of params
        row_highpass_neural_boxes.setColumnMinimumWidth(0, 100)
        row_highpass_neural_boxes.setColumnMinimumWidth(1, 100)
        
        # Param: highpass_neural_scope_yrange_mV
        self.line_edit_highpass_neural_scope_yrange_uV = PyQt5.QtWidgets.QLineEdit(    
            str(self.highpass_neural_scope_yrange_uV))
        self.line_edit_highpass_neural_scope_yrange_uV.returnPressed.connect(
            self.line_edit_highpass_neural_scope_yrange_uV_update)
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('highpass scope yrange (uV)'), 0, 0)
        row_highpass_neural_boxes.addWidget(
            self.line_edit_highpass_neural_scope_yrange_uV, 0, 1)

        # Fix the width of the grid of params
        row_highpass_neural_boxes.setColumnMinimumWidth(0, 100)
        row_highpass_neural_boxes.setColumnMinimumWidth(1, 100)
        
        # Param: checkbox for plot_ch0
        self.checkbox_plot_ch0_highpass = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch0_highpass.setChecked(True)
        self.checkbox_plot_ch0_highpass.stateChanged.connect(
            self.checkbox_plot_ch0_highpass_update)
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 0?'), 1, 0)
        row_highpass_neural_boxes.addWidget(self.checkbox_plot_ch0_highpass, 1, 1)
        
        # Param: checkbox for plot_ch2
        self.checkbox_plot_ch2_highpass = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch2_highpass.setChecked(True)
        self.checkbox_plot_ch2_highpass.stateChanged.connect(
            self.checkbox_plot_ch2_highpass_update)
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 2?'), 2, 0)
        row_highpass_neural_boxes.addWidget(self.checkbox_plot_ch2_highpass, 2, 1)
        
        # Param: checkbox for plot_ch4
        self.checkbox_plot_ch4_highpass = PyQt5.QtWidgets.QCheckBox()
        self.checkbox_plot_ch4_highpass.setChecked(True)
        self.checkbox_plot_ch4_highpass.stateChanged.connect(
            self.checkbox_plot_ch4_highpass_update)
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('plot ch 4?'), 3, 0)
        row_highpass_neural_boxes.addWidget(self.checkbox_plot_ch4_highpass, 3, 1)
        
        # Label: heart rate
        self.label_heart_rate = PyQt5.QtWidgets.QLabel('')
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('heart rate'), 4, 0)
        row_highpass_neural_boxes.addWidget(self.label_heart_rate, 4, 1)

        # Label: rms on each channel
        self.label_highpass_neural_rms = PyQt5.QtWidgets.QLabel('')
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('ABR band rms (uV): '), 5, 0)
        row_highpass_neural_boxes.addWidget(self.label_highpass_neural_rms, 5, 1)


        ## Fourth row: a row of audio widgets
        # Horizontal: plot widget, and then Grid of params
        row_audio_plot = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(row_audio_plot)
        row_audio_boxes = PyQt5.QtWidgets.QGridLayout()
        row_audio_plot.addWidget(self.speaker_plot_widget)
        row_audio_plot.addLayout(row_audio_boxes)

        # Fix the width of the grid of params
        row_audio_boxes.setColumnMinimumWidth(0, 100)
        row_audio_boxes.setColumnMinimumWidth(1, 100)
        
        # Param: audio_scope_xrange_s
        self.line_edit_audio_scope_xrange_s = PyQt5.QtWidgets.QLineEdit(
            str(self.audio_scope_xrange_s))
        self.line_edit_audio_scope_xrange_s.returnPressed.connect(
            self.line_edit_audio_scope_xrange_s_update)
        row_audio_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('audio scope xrange (s)'), 1, 0)
        row_audio_boxes.addWidget(self.line_edit_audio_scope_xrange_s, 1, 1)
        
        # Param: audio_scope_yrange_mV
        self.line_edit_audio_scope_yrange_mV = PyQt5.QtWidgets.QLineEdit(    
            str(self.audio_scope_yrange_mV))
        self.line_edit_audio_scope_yrange_mV.returnPressed.connect(
            self.line_edit_audio_scope_yrange_mV_update)
        row_audio_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('audio scope yrange (mV)'), 0, 0)
        row_audio_boxes.addWidget(self.line_edit_audio_scope_yrange_mV, 0, 1)        

        
        ## Fifth row: clicks and PSD readouts
        # Create and add layout for a horizontal row of widgets
        self.click_layout = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(self.click_layout)
        
        # Add widgets
        self.click_layout.addWidget(self.psd_monitor_widget)
        self.click_layout.addWidget(self.abr_pos_audio_monitor_widget)
        self.click_layout.addWidget(self.abr_neg_audio_monitor_widget)
        
        # GridLayout for params
        click_layout_grid = PyQt5.QtWidgets.QGridLayout()
        self.click_layout.addLayout(click_layout_grid)

        # Fix the width of the grid of params
        click_layout_grid.setColumnMinimumWidth(0, 100)
        click_layout_grid.setColumnMinimumWidth(1, 100)
        
        # Param: audio_scope_xrange_s
        self.line_edit_abr_audio_monitor_yrange_uV = (
            PyQt5.QtWidgets.QLineEdit(str(self.abr_audio_monitor_yrange_uV)))
        self.line_edit_abr_audio_monitor_yrange_uV.returnPressed.connect(
            self.line_edit_abr_audio_monitor_yrange_uV_update)
        click_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('clicks yrange (uV)'), 0, 0)
        click_layout_grid.addWidget(
            self.line_edit_abr_audio_monitor_yrange_uV, 0, 1)        

        # Label: plot time required
        self.label_plot_time_required = PyQt5.QtWidgets.QLabel('')
        click_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('plotting time required (ms)'), 1, 0)
        click_layout_grid.addWidget(
            self.label_plot_time_required, 1, 1)            
        
        
        ## Fourth row: ABR readouts
        # The bottom of the layout is horizontal
        self.abr_layout = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(self.abr_layout)
        
        # Add widgets
        self.abr_layout.addWidget(self.abr_neural_ch0_monitor_widget)
        self.abr_layout.addWidget(self.abr_neural_ch2_monitor_widget)
        self.abr_layout.addWidget(self.abr_neural_ch4_monitor_widget)

        # GridLayout for params
        abr_layout_grid = PyQt5.QtWidgets.QGridLayout()
        self.abr_layout.addLayout(abr_layout_grid)
        
        # Param: abr_neural_yrange_uV
        self.line_edit_abr_neural_yrange_uV = (
            PyQt5.QtWidgets.QLineEdit(str(self.abr_neural_yrange_uV)))
        self.line_edit_abr_neural_yrange_uV.returnPressed.connect(
            self.line_edit_abr_neural_yrange_uV_update)
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('abr yrange (uV)'), 1, 0)
        abr_layout_grid.addWidget(
            self.line_edit_abr_neural_yrange_uV, 1, 1)        

        # Label: ch0 abr noise
        self.label_abr_ch0_noise = PyQt5.QtWidgets.QLabel('')
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('ch0 ABR noise'), 2, 0)
        abr_layout_grid.addWidget(
            self.label_abr_ch0_noise, 2, 1)        

        # Label: ch2 abr noise
        self.label_abr_ch2_noise = PyQt5.QtWidgets.QLabel('')
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('ch2 ABR noise'), 3, 0)
        abr_layout_grid.addWidget(
            self.label_abr_ch2_noise, 3, 1)        

        # Label: ch4 abr noise
        self.label_abr_ch4_noise = PyQt5.QtWidgets.QLabel('')
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('ch4 ABR noise'), 4, 0)
        abr_layout_grid.addWidget(
            self.label_abr_ch4_noise, 4, 1)        

        
        ## Set labels and colors of `plot_widget`
        self.setup_plot_graphics()
       
        # Plots line_of_current_time and line
        self.initalize_plot_handles()

    def checkbox_plot_ch0_neural_update(self):
        checked = self.checkbox_plot_ch0_neural.checkState()
        if checked:
            self.neural_plot_handle_l[0].setVisible(True)
        else:
            self.neural_plot_handle_l[0].setVisible(False)

    def checkbox_plot_ch2_neural_update(self):
        """Set visibility of ch2 on neural plot based on checkbox"""
        # TODO: This hardcodes ch2 as index 1, fix
        checked = self.checkbox_plot_ch2_neural.checkState()
        if checked:
            self.neural_plot_handle_l[1].setVisible(True)
        else:
            self.neural_plot_handle_l[1].setVisible(False)

    def checkbox_plot_ch4_neural_update(self):
        """Set visibility of ch4 on neural plot based on checkbox"""
        # TODO: This hardcodes ch4 as index 2, fix
        checked = self.checkbox_plot_ch4_neural.checkState()
        if checked:
            self.neural_plot_handle_l[2].setVisible(True)
        else:
            self.neural_plot_handle_l[2].setVisible(False)

    def checkbox_plot_ch0_highpass_update(self):
        checked = self.checkbox_plot_ch0_highpass.checkState()
        if checked:
            self.highpass_neural_plot_handle_l[0].setVisible(True)
        else:
            self.highpass_neural_plot_handle_l[0].setVisible(False)

    def checkbox_plot_ch2_highpass_update(self):
        """Set visibility of ch2 on neural plot based on checkbox"""
        # TODO: This hardcodes ch2 as index 1, fix
        checked = self.checkbox_plot_ch2_highpass.checkState()
        if checked:
            self.highpass_neural_plot_handle_l[1].setVisible(True)
        else:
            self.highpass_neural_plot_handle_l[1].setVisible(False)

    def checkbox_plot_ch4_highpass_update(self):
        """Set visibility of ch4 on neural plot based on checkbox"""
        # TODO: This hardcodes ch4 as index 2, fix
        checked = self.checkbox_plot_ch4_highpass.checkState()
        if checked:
            self.highpass_neural_plot_handle_l[2].setVisible(True)
        else:
            self.highpass_neural_plot_handle_l[2].setVisible(False)

    def line_edit_neural_scope_xrange_s_update(self):
        try:
            text = self.line_edit_neural_scope_xrange_s.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.neural_scope_xrange_s = value

    def line_edit_neural_scope_yrange_mV_update(self):
        try:
            text = self.line_edit_neural_scope_yrange_mV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.neural_scope_yrange_mV = value

    def line_edit_highpass_neural_scope_yrange_uV_update(self):
        try:
            text = self.line_edit_highpass_neural_scope_yrange_uV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.highpass_neural_scope_yrange_uV = value

    def line_edit_audio_scope_xrange_s_update(self):
        try:
            text = self.line_edit_audio_scope_xrange_s.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.audio_scope_xrange_s = value

    def line_edit_audio_scope_yrange_mV_update(self):
        try:
            text = self.line_edit_audio_scope_yrange_mV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.audio_scope_yrange_mV = value
    
    def line_edit_abr_audio_monitor_yrange_uV_update(self):
        try:
            text = self.line_edit_abr_audio_monitor_yrange_uV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.abr_audio_monitor_yrange_uV = value    

    def line_edit_abr_neural_yrange_uV_update(self):
        try:
            text = self.line_edit_abr_neural_yrange_uV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.abr_neural_yrange_uV = value        
    
    def setup_plot_graphics(self):
        """Sets colors and labels of plot_widget
        
        Flow
        * Sets background to black and font to white
        * Sets title and axis labels
        * Adds a grid
        * Sets y-limits to [1, 9]
        """
        # Set the background of the plot to be black. Use 'w' for white
        self.neural_plot_widget.setBackground('k') 
        self.highpass_neural_plot_widget.setBackground('k') 
        self.speaker_plot_widget.setBackground('k') 
        self.abr_pos_audio_monitor_widget.setBackground('k')
        self.abr_neg_audio_monitor_widget.setBackground('k')
        self.abr_neural_ch0_monitor_widget.setBackground('k')
        self.abr_neural_ch2_monitor_widget.setBackground('k')
        self.abr_neural_ch4_monitor_widget.setBackground('k')

        # Set the title
        self.abr_pos_audio_monitor_widget.setTitle('positive clicks')
        self.abr_neg_audio_monitor_widget.setTitle('negative clicks')
        self.abr_neural_ch0_monitor_widget.setTitle('ch0 ABR')
        self.abr_neural_ch2_monitor_widget.setTitle('ch2 ABR')
        self.abr_neural_ch4_monitor_widget.setTitle('ch4 ABR')
        
        # Set the ylabel
        self.neural_plot_widget.setLabel('left', 'neural signal (mV)')
        self.highpass_neural_plot_widget.setLabel('left', 'highpass signal (uV)')
        self.speaker_plot_widget.setLabel('left', 'speaker signal (mV)')
        
        # Set the xlabel
        self.neural_plot_widget.setLabel('bottom', 'time (sec)')
        self.speaker_plot_widget.setLabel('bottom', 'time (sec)')
        
        # Set the range for the X axis
        self.neural_plot_widget.setXRange(
            -self.neural_scope_xrange_s, 0)
        self.highpass_neural_plot_widget.setXRange(
            -self.neural_scope_xrange_s, 0)
        self.speaker_plot_widget.setXRange(
            -self.audio_scope_xrange_s, 0)
        
        # Set the range for the Y axis
        self.neural_plot_widget.setYRange(
            -self.neural_scope_yrange_mV, self.neural_scope_yrange_mV)
        self.highpass_neural_plot_widget.setYRange(
            -self.highpass_neural_scope_yrange_uV, 
            self.highpass_neural_scope_yrange_uV)
        self.speaker_plot_widget.setYRange(
            -self.audio_scope_yrange_mV, self.audio_scope_yrange_mV)
        self.abr_pos_audio_monitor_widget.setXRange(
            -self.audio_extract_win_samples, self.audio_extract_win_samples)
        self.abr_pos_audio_monitor_widget.setYRange(
            0, self.abr_audio_monitor_yrange_uV) # abslog scale
        self.abr_neg_audio_monitor_widget.setXRange(
            -self.audio_extract_win_samples, self.audio_extract_win_samples)
        self.abr_neg_audio_monitor_widget.setYRange(
            0, self.abr_audio_monitor_yrange_uV) # abslog scale
        self.abr_neural_ch0_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
        self.abr_neural_ch2_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
        self.abr_neural_ch4_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
    
        # PSD plot
        self.psd_monitor_widget.setLogMode(True, True)
        self.psd_monitor_widget.setXRange(0, 4)
        self.psd_monitor_widget.setYRange(-5, 5)
    
    def initalize_plot_handles(self):
        """Plots line_of_current_time and line
        
        Creates these handles:
            self.line_of_current_time : a line that moves with the current time
            self.plot_handle_unrewarded_pokes : a raster plot of unrewarded
                pokes in red
            self.plot_handle_rewarded_incorrect_pokes : a raster plot of 
                rewarded incorrect pokes in blue
            self.plot_handle_rewarded_correct_pokes : a raster plot of 
                rewarded correct pokes in green
        """
        # Plot max lines at min and max in the neural_plot
        self.neural_plot_widget.addLine(x=None, y=187.5, pen={'color': 'w'})
        self.neural_plot_widget.addLine(x=None, y=-187.5, pen={'color': 'w'})
        
        # Set up each handle for the neural plot
        self.neural_plot_handle_l = []
        for n_channel, channel in enumerate(self.neural_channels_to_plot):
            handle = self.neural_plot_widget.plot(
                x=[], y=[], 
                pen=(n_channel, len(self.neural_channels_to_plot))
                )
            handle.setAlpha(0.5, auto=False)
            self.neural_plot_handle_l.append(handle)
        
        # Set up each handle for the highpass neural plot
        self.highpass_neural_plot_handle_l = []
        for n_channel, channel in enumerate(self.neural_channels_to_plot):
            handle = self.highpass_neural_plot_widget.plot(
                x=[], y=[], 
                pen=(n_channel, len(self.neural_channels_to_plot))
                )
            handle.setAlpha(0.5, auto=False)            
            self.highpass_neural_plot_handle_l.append(handle)
        
        # Heartbeat plot
        self.heartbeat_plot_handle = self.highpass_neural_plot_widget.plot(
            x=[], y=[],
            pen=None,
            symbol='o',
            symbolPen='w',
            symbolBrush=None,
            )
        
        # Set up speaker handle
        self.speaker_plot_handle = self.speaker_plot_widget.plot(x=[], y=[])

        # PSD widget
        self.psd_monitor_widget_handle_l = []
        for n_channel in range(3):
            handle = self.psd_monitor_widget.plot(
                x=[], y=[],
                pen=(n_channel, len(self.neural_channels_to_plot))
                )
            handle.setAlpha(0.5, auto=False)
            self.psd_monitor_widget_handle_l.append(handle)
    
        # Add a line for each
        self.abr_audio_monitor_pos_handle_l = []
        self.abr_audio_monitor_neg_handle_l = []
        self.abr_ch0_handle_l = []
        self.abr_ch2_handle_l = []
        self.abr_ch4_handle_l = []
        for n_amplitude, amplitude_label in enumerate(self.amplitude_labels):
            # Positive clicks
            handle = self.abr_pos_audio_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_audio_monitor_pos_handle_l.append(handle)

            # Negative clicks
            handle = self.abr_neg_audio_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_audio_monitor_neg_handle_l.append(handle)

            # ABRs ch0
            handle = self.abr_neural_ch0_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_ch0_handle_l.append(handle)

            # ABRs ch2
            handle = self.abr_neural_ch2_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_ch2_handle_l.append(handle)

            # ABRs ch4
            handle = self.abr_neural_ch4_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_ch4_handle_l.append(handle)
    
    def start(self):
        # Start the timer that will continuously update
        self.timer_update_plot.start(self.update_interval_ms)  

    def stop(self):
        """Deactivates plot updates"""
        self.timer_update_plot.stop()

    def get_data(self):
        """Get data from self.abr_device
        
        TODO: consider moving this to ABR_Device
        
        duration_to_get_ms : numeric
            Get this many milliseconds of recent data
        
        Returns: big_data, headers_df, t_values
            None if no data available
            big_data : data in uV
            headers_df : pandas.DataFrame
            t_values : time of each sample in seconds since acquisition start
            
            or returns None, None, None if there is no data to get
        """
        # How many chunks we need in order to fill out the plot
        needed_chunks = int(np.ceil(
            self.duration_data_to_analyze_s * self.abr_device.sampling_rate / 500
            ))
        
        # We can't get more data than there is available
        # The -1 keeps us from ever getting the most recent one
        n_chunks_available = len(self.abr_device.deq_data) - 1
        if needed_chunks > n_chunks_available:
            needed_chunks = n_chunks_available
        
        # Return if no data available
        if needed_chunks <= 0:
            return None, None, None
        
        # If data is added to the right during this operation, it won't matter
        # because the index is still valid. But if data is also emptied from
        # the left, the data will tear. Fortunately emptying from the left
        # is more rare.
        # TODO: this deq is no longer emptied by anyone. Faster way than
        # re-reading the whole thing all the time?
        data_chunk_l = []
        data_header_l = []
        for idx in range(n_chunks_available - needed_chunks, n_chunks_available):
            data_chunk = self.abr_device.deq_data[idx]
            data_header = self.abr_device.deq_header[idx]
            
            # Store
            data_chunk_l.append(data_chunk)
            data_header_l.append(data_header)
        
        # Concat the data
        big_data = np.concatenate(data_chunk_l)
        headers_df = pandas.DataFrame.from_records(data_header_l)

        
        ## Get data in real physical units
        # Convert to uV (full-scale range is 9V)
        big_data = big_data * 9e6 / 2 ** 24
        
        # Account for gain (TODO: load from config)
        # Note: maximum achievable value is 4.5V / gain, or 0.1875 V for gain=24
        big_data = big_data / np.array(self.abr_device.gains)
        
        # Use headers_df to make the xvals
        # If there are dropped packets, these xvals will be wrong, but I 
        # don't think it really matters or is worth fixing here
        packet_numbers = np.unwrap(headers_df['packet_num'], period=256)
        start_time_samples = packet_numbers[0] * 500
        stop_time_samples = (packet_numbers[-1] + 1) * 500
        t_values = (
            np.arange(start_time_samples, stop_time_samples) / 
            self.abr_device.sampling_rate
            )

        return big_data, headers_df, t_values

    def extract_audio_onsets(self, speaker_channel):
        """Highpass filter, extract onsets, and classify clicks
        
        Return: audio_data_hp, click_params
        """
        ## Extract onsets
        # Highpass audio data above 50 Hz to remove noise
        # This remove some baseline noise, but also makes the big clicks "ring"
        nyquist_freq = self.abr_device.sampling_rate / 2
        ad_ahi, ad_bhi = scipy.signal.butter(
            2, (50 / nyquist_freq), btype='highpass')
        audio_data_hp = scipy.signal.lfilter(ad_ahi, ad_bhi, speaker_channel)

        # Extract onsets
        onsets = paclab.abr.abr.get_single_onsets(
            audio_data_hp, 
            audio_threshold=10**self.amplitude_cuts[0],
            abr_start_sample=-40, 
            abr_stop_sample=120,
            )

        # DataFrame them
        click_params = pandas.Series(onsets, name='t_samples').to_frame()


        ## Categorize the onsets
        # Extract the amplitude of each click
        click_params['amplitude_V'] = audio_data_hp[onsets]
        click_params['amplitude_log_abs'] = np.log10(
            np.abs(click_params['amplitude_V']))

        # Cut
        click_params['amplitude_idx'] = pandas.cut(
            click_params['amplitude_log_abs'], 
            bins=self.amplitude_cuts, labels=False)

        # Drop any that are above the last cut
        big_click_mask = (
            click_params['amplitude_log_abs'] > self.amplitude_cuts[-1])
        if np.any(big_click_mask):
            # TODO: warn here, but just once
            click_params = click_params[~big_click_mask]
        
        # After this, there should be no clicks with null amplitude_idx
        click_params['amplitude_idx'] = click_params['amplitude_idx'].astype(int)

        # Label
        click_params['amplitude'] = self.amplitude_labels[
            click_params['amplitude_idx']]

        # Define polarity
        click_params['polarity'] = click_params['amplitude_V'] > 0
    
        return audio_data_hp, click_params

    def update(self, abr_start_sample=-40, abr_stop_sample=120):
        """Update all widgets with recent data
        
        Workflow
        * Get data from abr_device
        * Update oscilloscope
        * Extract onsets
        * Lock audio and neural data onto those onsets
        * Plot those locked data
        """
        ## Debug timing
        times = []
        times.append(('start', datetime.datetime.now()))
        
        
        ## Update graphical elements that may have changed
        self.setup_plot_graphics()
        times.append(('setup_plot_graphics done', datetime.datetime.now()))
        
        
        ## Get data (now in uV)
        # This takes a while - 40 ms max
        big_data, headers_df, t_values = self.get_data()
        
        times.append(('data gotten 1', datetime.datetime.now()))
        
        # Continue if not enough
        if big_data is None:
            if self.verbose:
                print('waiting to receive enough data to plot')
            return

        # Store the size of data received
        self.label_analyze_data_duration_s.setText('{:.1f}'.format(
            len(big_data) / self.abr_device.sampling_rate))

        # Show only connected channels (TODO: allow user to specify)
        neural_data = big_data[:, self.neural_channels_to_plot]
        
        # Parse out the audio channel (TODO: allow user to specify)
        speaker_channel = big_data[:, self.speaker_channel]
        
        times.append(('data gotten 2', datetime.datetime.now()))
        
        
        ## Bandpass neural
        # Each of these filters can take 20 ms * n_channels
        nyquist_freq = self.abr_device.sampling_rate / 2
        ahi, bhi = scipy.signal.butter(
            2, (
            self.abr_highpass_freq / nyquist_freq, 
            self.abr_lowpass_freq / nyquist_freq), 
            btype='bandpass')
        neural_data_hp = scipy.signal.filtfilt(ahi, bhi, neural_data, axis=0)

        times.append(('first hp', datetime.datetime.now()))

        
        ## Bandpass heartbeat
        # TODO: hardcoded to ch0, make this selectable
        nyquist_freq = self.abr_device.sampling_rate / 2
        ahi, bhi = scipy.signal.butter(
            2, (
            self.heartbeat_highpass_freq / nyquist_freq, 
            self.heartbeat_lowpass_freq / nyquist_freq), 
            btype='bandpass')
        ekg_signal = scipy.signal.filtfilt(ahi, bhi, neural_data[:, 0])

        times.append(('second hp', datetime.datetime.now()))

        # Find heartbeats
        # These are indexed into ekg_signal
        self.heartbeats = scipy.signal.find_peaks(
            ekg_signal, 60, distance=100)[0]
        
        # Include only those heartbeats in the recency window
        ekg_recent_duration_window_samples = (
            self.ekg_recent_duration_window_s * self.abr_device.sampling_rate)
        self.heartbeats = self.heartbeats[
            self.heartbeats > 
            (len(ekg_signal) - ekg_recent_duration_window_samples)
            ]
        
        times.append(('findpeaks', datetime.datetime.now()))
        
        # Determine how much time is included in the recency window
        len_recency_window_samples = np.min([
            ekg_recent_duration_window_samples, len(ekg_signal)])
        self.heart_rate_bpm = (
            len(self.heartbeats) / len_recency_window_samples * 
            self.abr_device.sampling_rate * 60)
        self.label_heart_rate.setText(str(int(np.rint(self.heart_rate_bpm))))
        
        times.append(('heartbeat processed', datetime.datetime.now()))
        
        
        ## Set labels for RMS power in each band
        # Set the label for the raw rms power
        raw_rms = neural_data.std(axis=0) / 1e3
        raw_rms_s = ' '.join([
            'ch{}={:.1f};'.format(chan, val) for chan, val in 
            zip(self.neural_channels_to_plot, raw_rms)])
        self.label_raw_neural_rms.setText(raw_rms_s)

        # Set the label for the hp rms power
        hp_rms = neural_data_hp.std(axis=0)
        hp_rms_s = ' '.join([
            'ch{}={:.1f};'.format(chan, val) for chan, val in 
            zip(self.neural_channels_to_plot, hp_rms)])
        self.label_highpass_neural_rms.setText(hp_rms_s)
        
        
        ## Update plot with the new xvals and yvals
        # This plotting block takes 20 ms
        # Use relative time here
        xvals = np.arange(-len(big_data), 0) / self.abr_device.sampling_rate
        
        # Include only the time that we need
        # This is important for efficiency
        include_mask = xvals > -self.neural_scope_xrange_s
        
        # Set each neural plot
        for n_channel, neural_channel in enumerate(neural_data.T):
            self.neural_plot_handle_l[n_channel].setData(
                x=xvals[include_mask], y=neural_channel[include_mask] / 1e3)
        
        # Set each highpass neural plot
        for n_channel, neural_channel in enumerate(neural_data_hp.T):
            self.highpass_neural_plot_handle_l[n_channel].setData(
                x=xvals[include_mask], y=neural_channel[include_mask])
        
        # Plot the heartbeats
        # TODO: make this more efficient
        self.heartbeat_plot_handle.setData(
            x=xvals[self.heartbeats],
            y=neural_data_hp[self.heartbeats, 0],
            )
        
        # Set speaker plot
        self.speaker_plot_handle.setData(x=xvals, y=speaker_channel / 1e3)
    
        times.append(('neural plots updated', datetime.datetime.now()))


        ## Plot PSD
        Pxx_l = []
        for ncol, col in enumerate(neural_data.T):
            # Get handle
            handle = self.psd_monitor_widget_handle_l[ncol]
            
            # Take only the last 4 s of data
            to_psd = col[-2**16:]
            
            # Data is in V
            Pxx, freqs = psd(
                to_psd, 
                NFFT=16384, 
                Fs=self.abr_device.sampling_rate,
                noverlap=8192,
                )
            
            # When it's railed, Pxx will be zero everywhere
            Pxx = Pxx + 1e-4
            
            # Plot
            handle.setData(
                x=freqs,
                y=Pxx,
                )


        ## Extract onsets and click params
        audio_data_hp, click_params = self.extract_audio_onsets(speaker_channel)
        
        # Drop onsets too close to the edges
        click_params = click_params[
            (click_params['t_samples'] > self.audio_extract_win_samples) &
            (click_params['t_samples'] < 
            len(audio_data_hp) - self.audio_extract_win_samples)
            ]
        
        times.append(('clicks extracted', datetime.datetime.now()))
        
        
        ## These parts are done only if there are clicks to analyze
        if len(click_params) > 0:
        
            ## Extract each trigger from the audio signal
            # A 0.1 ms click is only a few samples long
            triggered_ad = np.array([
                audio_data_hp[
                trigger - self.audio_extract_win_samples:
                trigger + self.audio_extract_win_samples]
                for trigger in click_params['t_samples']])

            # DataFrame
            triggered_ad = pandas.DataFrame(triggered_ad)
            triggered_ad.columns = pandas.Series(range(
                -self.audio_extract_win_samples, 
                self.audio_extract_win_samples),
                name='timepoint')
            triggered_ad.index = pandas.MultiIndex.from_frame(
                click_params[['amplitude', 'polarity', 't_samples']])
            triggered_ad = triggered_ad.reorder_levels(
                ['amplitude', 'polarity', 't_samples']).sort_index()
            
            times.append(('audio triggered', datetime.datetime.now()))


            ## Extract neural data locked to onsets
            # Extract highpassed neural data around triggers
            # Shape is (n_triggers, n_timepoints, n_channels)
            triggered_neural = np.array([
                neural_data_hp[trigger + abr_start_sample:trigger + abr_stop_sample]
                for trigger in click_params['t_samples']])
            
            # Remove channel as a level
            triggered_neural = triggered_neural.reshape(
                [triggered_neural.shape[0], -1])        

            # DataFrame
            triggered_neural = pandas.DataFrame(triggered_neural)
            triggered_neural.index = pandas.MultiIndex.from_frame(
                click_params[['amplitude', 'polarity', 't_samples']])
            triggered_neural = triggered_neural.reorder_levels(
                ['amplitude', 'polarity', 't_samples']).sort_index()

            # The columns are interdigitated samples and channels
            triggered_neural.columns = pandas.MultiIndex.from_product([
                pandas.Index(range(abr_start_sample, abr_stop_sample), name='timepoint'),
                pandas.Index(self.neural_channels_to_plot, name='channel')
                ])

            # Put channels first
            triggered_neural = triggered_neural.swaplevel(axis=1).sort_index(axis=1)        

            times.append(('neural triggered', datetime.datetime.now()))


            ## Aggregate
            # Average by condition
            avg_by_condition = triggered_neural.groupby(
                ['polarity', 'amplitude']).mean()

            # Average out polarity
            avg_abrs = avg_by_condition.groupby('amplitude').mean()

            # Average audio by condition
            avg_audio_by_condition = triggered_ad.groupby(['amplitude', 'polarity']).mean()
            times.append(('aggregation done', datetime.datetime.now()))


            ## Estimate noise level
            # We estimate noise for each amplitude separately, and then mean
            # them together.
            ch0_noise = avg_abrs.loc[:, 0].loc[:, -40:-19].std(axis=1).mean()
            ch2_noise = avg_abrs.loc[:, 2].loc[:, -40:-19].std(axis=1).mean()
            ch4_noise = avg_abrs.loc[:, 4].loc[:, -40:-19].std(axis=1).mean()

            self.label_abr_ch0_noise.setText('{:.2f} uV'.format(ch0_noise))
            self.label_abr_ch2_noise.setText('{:.2f} uV'.format(ch2_noise))
            self.label_abr_ch4_noise.setText('{:.2f} uV'.format(ch4_noise))
            
            
            ## Plot clicks by amplitude
            # First abslog avg_audio_by_condition for display
            avg_audio_by_condition = np.log10(np.abs(avg_audio_by_condition))
            avg_audio_by_condition[avg_audio_by_condition < 0] = 0
            
            # Reindex by amplitudes that should exist so things line up
            avg_audio_by_condition = avg_audio_by_condition.reindex(
                triggered_ad.index.levels[0], level='amplitude')
            
            # Separate negatives and positives
            # Can be None if we haven't had any yet (eg, at beginning)
            try:
                neg_clicks = avg_audio_by_condition.xs(False, level='polarity')
            except KeyError:
                neg_clicks = None
            
            try:
                pos_clicks = avg_audio_by_condition.xs(True, level='polarity')
            except KeyError:
                pos_clicks = None

            # Plot negatives
            if neg_clicks is not None:
                zobj = zip(self.abr_audio_monitor_neg_handle_l, neg_clicks.values)
                for handle, topl in zobj:
                    handle.setData(
                        x=avg_audio_by_condition.columns.values, y=topl)

            # Plot positives
            if pos_clicks is not None:
                zobj = zip(self.abr_audio_monitor_pos_handle_l, pos_clicks.values)
                for handle, topl in zobj:
                    handle.setData(
                        x=avg_audio_by_condition.columns.values, y=topl)
            
            
            ## Plot ABR by amplitude
            # Reindex by amplitudes that should exist so things line up
            avg_abrs = avg_abrs.reindex(
                triggered_ad.index.levels[0], level='amplitude')
            
            # Plot ch0
            abr_ch0 = avg_abrs.loc[:, 0]
            zobj = zip(self.abr_ch0_handle_l, abr_ch0.values)
            for handle, topl in zobj:
                handle.setData(x=abr_ch0.columns.values, y=topl)

            # Plot ch2
            abr_ch2 = avg_abrs.loc[:, 2]
            zobj = zip(self.abr_ch2_handle_l, abr_ch2.values)
            for handle, topl in zobj:
                handle.setData(x=abr_ch2.columns.values, y=topl)

            # Plot ch4
            abr_ch4 = avg_abrs.loc[:, 4]
            zobj = zip(self.abr_ch4_handle_l, abr_ch4.values)
            for handle, topl in zobj:
                handle.setData(x=abr_ch4.columns.values, y=topl)

        
        ## Print debug timing information
        times.append(('done', datetime.datetime.now()))
        times = pandas.DataFrame.from_records(times, columns=['event', 't'])
        times['diff'] = times['t'].diff().dt.total_seconds()
        
        # Store total time taken
        self.update_time_taken = times['diff'].sum()
        self.label_plot_time_required.setText(
            '{:.1f} ms'.format(self.update_time_taken * 1000))
        
        # Average over times
        self.times_l.append(times)
        concatted = pandas.concat(
            self.times_l[-20:], 
            keys=range(len(self.times_l[-20:])), 
            names=['rep'])
        meaned = concatted.groupby('event')['diff'].mean()

        # More verbose output
        if self.print_timing_information:
            print(meaned)
            print(meaned.sum())

