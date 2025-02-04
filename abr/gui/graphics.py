"""Objects to run the ABR GUI

Impedance measurement
Plot lines at max so we can see railing, or flag when it's totally out of range
Display total time taken per update call
Make ABR plots bigger
Allow flipping live switch
efficiency updates in update
invert color order in ABR plot
make a mean ABR
"""

import sys
import traceback
import datetime
import os
import scipy.signal
import paclab.abr
import numpy as np
import pandas
import PyQt5.QtWidgets
import PyQt5.QtCore
import pyqtgraph as pg
from . import ABR_Device

# Temporary workaround
# In the main branch, this was just paclab.abr
# Now everything's been moved into a subfolder
# In any case we need to remove all references to paclab.abr.abr and 
# paclab.abr.abr_gui
import paclab.abr.abr


class OscilloscopeWidget(PyQt5.QtWidgets.QWidget):
    def __init__(self, 
        abr_device, 
        update_interval_ms=250, # it can't really go any faster
        duration_data_to_analyze_s=300, 
        neural_scope_xrange_s=5,
        neural_scope_yrange_uV=30000,
        highpass_neural_scope_yrange_uV=300,
        audio_scope_xrange_s=5,
        audio_scope_yrange_uV=300000,
        abr_audio_monitor_yrange_uV=5, # abslog scale
        abr_neural_yrange_uV=5,
        audio_extract_win_samples=10,
        *args, **kwargs):
        
        ## Superclass PyQt5.QtWidgets.QWidget init
        super().__init__(*args, **kwargs)
        
        
        ## Instance variables
        # abr_device, where the data comes from
        self.abr_device = abr_device
        
        # Timers for continuous updating
        # Create a PyQt5.QtCore.QTimer object to continuously update the plot         
        self.timer_update_plot = PyQt5.QtCore.QTimer(self) 
        self.timer_update_plot.timeout.connect(self.update)  
        self.update_interval_ms = update_interval_ms

        # Parameters that cannot be set by user interaction
        self.abr_highpass_freq = 300
        self.abr_lowpass_freq = 3000
        self.heartbeat_highpass_freq = 50
        self.heartbeat_lowpass_freq = 3000
        self.ekg_recent_duration_window_s = 30
        self.duration_data_to_analyze_s = duration_data_to_analyze_s
        self.audio_extract_win_samples = audio_extract_win_samples

        # TODO: get this from config
        self.neural_channels_to_plot = [0, 2]
        self.speaker_channel = 7

        # Parameters that can be set by user interaction
        self.neural_scope_xrange_s = neural_scope_xrange_s
        self.neural_scope_yrange_uV = neural_scope_yrange_uV
        self.highpass_neural_scope_yrange_uV = highpass_neural_scope_yrange_uV
        self.audio_scope_xrange_s = audio_scope_xrange_s
        self.audio_scope_yrange_uV = audio_scope_yrange_uV
        self.abr_audio_monitor_yrange_uV = abr_audio_monitor_yrange_uV
        self.abr_neural_yrange_uV = abr_neural_yrange_uV

        # Parameters that we can't set until we have data
        self.heart_rate = -1
        
        
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
        self.abr_pos_audio_monitor_widget = pg.PlotWidget()
        self.abr_neg_audio_monitor_widget = pg.PlotWidget()
        self.abr_neural_ch0_monitor_widget = pg.PlotWidget()
        self.abr_artefact_ch0_monitor_widget = pg.PlotWidget()
        self.abr_neural_ch2_monitor_widget = pg.PlotWidget()
        self.abr_artefact_ch2_monitor_widget = pg.PlotWidget()
        
        
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
        
        # Param: neural_scope_xrange_s
        self.line_edit_neural_scope_xrange_s = PyQt5.QtWidgets.QLineEdit(
            str(self.neural_scope_xrange_s))
        self.line_edit_neural_scope_xrange_s.returnPressed.connect(
            self.line_edit_neural_scope_xrange_s_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('neural scope xrange (s)'), 1, 0)
        row_neural_boxes.addWidget(self.line_edit_neural_scope_xrange_s, 1, 1)
        
        # Param: neural_scope_yrange_uV
        self.line_edit_neural_scope_yrange_uV = PyQt5.QtWidgets.QLineEdit(    
            str(self.neural_scope_yrange_uV))
        self.line_edit_neural_scope_yrange_uV.returnPressed.connect(
            self.line_edit_neural_scope_yrange_uV_update)
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('neural scope yrange (uV)'), 0, 0)
        row_neural_boxes.addWidget(self.line_edit_neural_scope_yrange_uV, 0, 1)

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

        # Param: label for amount of data received
        self.label_analyze_data_duration_s = PyQt5.QtWidgets.QLabel()
        row_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('data analyzed (s)'), 4, 0)
        row_neural_boxes.addWidget(self.label_analyze_data_duration_s, 4, 1)
        


        ## Second row: a row of highpass neural widgets
        # Horizontal: plot widget, and then Grid of params
        row_highpass_neural_plot = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(row_highpass_neural_plot)
        row_highpass_neural_boxes = PyQt5.QtWidgets.QGridLayout()
        row_highpass_neural_plot.addWidget(self.highpass_neural_plot_widget)
        row_highpass_neural_plot.addLayout(row_highpass_neural_boxes)
        
        # Param: highpass_neural_scope_yrange_uV
        self.line_edit_highpass_neural_scope_yrange_uV = PyQt5.QtWidgets.QLineEdit(    
            str(self.highpass_neural_scope_yrange_uV))
        self.line_edit_highpass_neural_scope_yrange_uV.returnPressed.connect(
            self.line_edit_highpass_neural_scope_yrange_uV_update)
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('highpass scope yrange (uV)'), 0, 0)
        row_highpass_neural_boxes.addWidget(
            self.line_edit_highpass_neural_scope_yrange_uV, 0, 1)
        
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
        
        # Param: heart rate
        self.label_heart_rate = PyQt5.QtWidgets.QLabel('')
        row_highpass_neural_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('heart rate'), 3, 0)
        row_highpass_neural_boxes.addWidget(self.label_heart_rate, 3, 1)


        ## Second row: a row of audio widgets
        # Horizontal: plot widget, and then Grid of params
        row_audio_plot = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(row_audio_plot)
        row_audio_boxes = PyQt5.QtWidgets.QGridLayout()
        row_audio_plot.addWidget(self.speaker_plot_widget)
        row_audio_plot.addLayout(row_audio_boxes)
        
        # Param: audio_scope_xrange_s
        self.line_edit_audio_scope_xrange_s = PyQt5.QtWidgets.QLineEdit(
            str(self.audio_scope_xrange_s))
        self.line_edit_audio_scope_xrange_s.returnPressed.connect(
            self.line_edit_audio_scope_xrange_s_update)
        row_audio_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('audio scope xrange (s)'), 1, 0)
        row_audio_boxes.addWidget(self.line_edit_audio_scope_xrange_s, 1, 1)
        
        # Param: audio_scope_yrange_uV
        self.line_edit_audio_scope_yrange_uV = PyQt5.QtWidgets.QLineEdit(    
            str(self.audio_scope_yrange_uV))
        self.line_edit_audio_scope_yrange_uV.returnPressed.connect(
            self.line_edit_audio_scope_yrange_uV_update)
        row_audio_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('audio scope yrange (uV)'), 0, 0)
        row_audio_boxes.addWidget(self.line_edit_audio_scope_yrange_uV, 0, 1)        

        
        ## Third row: ABR readouts
        # The bottom of the layout is horizontal
        self.abr_layout = PyQt5.QtWidgets.QHBoxLayout(self) 
        self.layout.addLayout(self.abr_layout)
        
        # Add widgets
        self.abr_layout.addWidget(self.abr_pos_audio_monitor_widget)
        self.abr_layout.addWidget(self.abr_neg_audio_monitor_widget)
        self.abr_layout.addWidget(self.abr_neural_ch0_monitor_widget)
        self.abr_layout.addWidget(self.abr_artefact_ch0_monitor_widget)
        self.abr_layout.addWidget(self.abr_neural_ch2_monitor_widget)
        self.abr_layout.addWidget(self.abr_artefact_ch2_monitor_widget)

        # GridLayout for params
        abr_layout_grid = PyQt5.QtWidgets.QGridLayout()
        self.abr_layout.addLayout(abr_layout_grid)
        
        # Param: audio_scope_xrange_s
        self.line_edit_abr_audio_monitor_yrange_uV = (
            PyQt5.QtWidgets.QLineEdit(str(self.abr_audio_monitor_yrange_uV)))
        self.line_edit_abr_audio_monitor_yrange_uV.returnPressed.connect(
            self.line_edit_abr_audio_monitor_yrange_uV_update)
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('clicks yrange (uV)'), 0, 0)
        abr_layout_grid.addWidget(
            self.line_edit_abr_audio_monitor_yrange_uV, 0, 1)        
        
        # Param: abr_neural_yrange_uV
        self.line_edit_abr_neural_yrange_uV = (
            PyQt5.QtWidgets.QLineEdit(str(self.abr_neural_yrange_uV)))
        self.line_edit_abr_neural_yrange_uV.returnPressed.connect(
            self.line_edit_abr_neural_yrange_uV_update)
        abr_layout_grid.addWidget(
            PyQt5.QtWidgets.QLabel('abr yrange (uV)'), 1, 0)
        abr_layout_grid.addWidget(
            self.line_edit_abr_neural_yrange_uV, 1, 1)        

        
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

    def line_edit_neural_scope_xrange_s_update(self):
        try:
            text = self.line_edit_neural_scope_xrange_s.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.neural_scope_xrange_s = value

    def line_edit_neural_scope_yrange_uV_update(self):
        try:
            text = self.line_edit_neural_scope_yrange_uV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.neural_scope_yrange_uV = value

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

    def line_edit_audio_scope_yrange_uV_update(self):
        try:
            text = self.line_edit_audio_scope_yrange_uV.text()
            value = float(text)
        except ValueError:
            print(f'warning: cannont convert {text} to float, ignoring')
            return
        
        self.audio_scope_yrange_uV = value
    
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
        self.abr_artefact_ch0_monitor_widget.setBackground('k')
        self.abr_neural_ch2_monitor_widget.setBackground('k')
        self.abr_artefact_ch2_monitor_widget.setBackground('k')

        # Set the title
        self.abr_pos_audio_monitor_widget.setTitle('positive clicks')
        self.abr_neg_audio_monitor_widget.setTitle('negative clicks')
        self.abr_neural_ch0_monitor_widget.setTitle('ch0 ABR')
        self.abr_artefact_ch0_monitor_widget.setTitle('ch0 artefact')
        self.abr_neural_ch2_monitor_widget.setTitle('ch2 ABR')
        self.abr_artefact_ch2_monitor_widget.setTitle('ch2 artefact')
        
        # Set the ylabel
        self.neural_plot_widget.setLabel('left', 'neural signal (uV)')
        self.highpass_neural_plot_widget.setLabel('left', 'highpass signal (uV)')
        self.speaker_plot_widget.setLabel('left', 'speaker signal (uV)')
        
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
            -self.neural_scope_yrange_uV, self.neural_scope_yrange_uV)
        self.highpass_neural_plot_widget.setYRange(
            -self.highpass_neural_scope_yrange_uV, 
            self.highpass_neural_scope_yrange_uV)
        self.speaker_plot_widget.setYRange(
            -self.audio_scope_yrange_uV, self.audio_scope_yrange_uV)
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
        self.abr_artefact_ch0_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
        self.abr_neural_ch2_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
        self.abr_artefact_ch2_monitor_widget.setYRange(
            -self.abr_neural_yrange_uV, self.abr_neural_yrange_uV)
    
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
    
        # Add a line for each
        self.abr_audio_monitor_pos_handle_l = []
        self.abr_audio_monitor_neg_handle_l = []
        self.abr_ch0_handle_l = []
        self.artefact_ch0_handle_l = []
        self.abr_ch2_handle_l = []
        self.artefact_ch2_handle_l = []
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

            # Artefacts ch0
            handle = self.abr_artefact_ch0_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.artefact_ch0_handle_l.append(handle)

            # ABRs ch2
            handle = self.abr_neural_ch2_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.abr_ch2_handle_l.append(handle)

            # Artefacts ch2
            handle = self.abr_artefact_ch2_monitor_widget.plot(
                x=[], y=[],
                pen=(n_amplitude, len(self.amplitude_labels))
                )
            self.artefact_ch2_handle_l.append(handle)
    
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
        #~ # How many chunks we need in order to fill out the plot
        #~ needed_chunks = int(np.ceil(
            #~ self.duration_data_to_analyze_s * self.abr_device.sampling_rate / 500
            #~ ))
        
        #~ # It's filled from the right by ThreadedSerialReader
        #~ # And emptied from the left by ThreadedFileWriter, which won't empty
        #~ #   it below a certain minimum length
        #~ # We can't get more than minimum deq_length
        #~ if self.abr_device.tfw is not None:
            #~ # This guard is needed because we don't have a dummy TFW yet
            #~ if needed_chunks > self.abr_device.tfw.minimum_deq_length:
                #~ needed_chunks = self.abr_device.tfw.minimum_deq_length

        #~ # We can't get more data than there is available
        #~ n_chunks_available = len(self.abr_device.tsr.deq_data)
        #~ if needed_chunks > n_chunks_available:
            #~ needed_chunks = n_chunks_available
        
        #~ # Return if no data available
        #~ if needed_chunks == 0:
            #~ return None, None, None
        
        #~ # If data is added to the right during this operation, it won't matter
        #~ # because the index is still valid. But if data is also emptied from
        #~ # the left, the data will tear. Fortunately emptying from the left
        #~ # is more rare.
        #~ # TODO: use a lock here to prevent that
        #~ data_chunk_l = []
        #~ data_header_l = []
        #~ for idx in range(n_chunks_available - needed_chunks, n_chunks_available):
            #~ data_chunk = self.abr_device.tsr.deq_data[idx]
            #~ data_header = self.abr_device.tsr.deq_headers[idx]
            
            #~ # Store
            #~ data_chunk_l.append(data_chunk)
            #~ data_header_l.append(data_header)
        
        #~ # Concat the data
        #~ big_data = np.concatenate(data_chunk_l)
        #~ headers_df = pandas.DataFrame.from_records(data_header_l)
        
        
        
        
        # Get from tfw
        if self.abr_device.tfw.big_data is None:
            return None, None, None
        
        big_data = self.abr_device.tfw.big_data[
            :self.abr_device.tfw.big_data_last_col]
        headers_df = pandas.DataFrame.from_records(self.abr_device.tfw.headers_l)
        
      
        
        
        
        
        # Convert to uV
        big_data = big_data * 9e6 / 2**24 # right?
        
        # Account for gain (TODO: load from config)
        big_data = big_data / np.array(self.abr_device.gains)
        #~ big_data[:, 0] = big_data[:, 0] / 24
        #~ big_data[:, 2] = big_data[:, 2] / 24
        
        # Use headers_df to make the xvals
        packet_numbers = np.unwrap(headers_df['packet_num'], period=256)
        start_time_samples = packet_numbers[0] * 500
        stop_time_samples = (packet_numbers[-1] + 1) * 500
        t_values = (
            np.arange(start_time_samples, stop_time_samples) / 
            self.abr_device.sampling_rate
            )
        
        # Check for tearing
        if (np.diff(packet_numbers) != 1).any():
            raise ValueError('data is torn!')   

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
        if big_data is None or len(big_data) < 1000:
            print('warning: no data received!')
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
                x=xvals[include_mask], y=neural_channel[include_mask])
        
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
        self.speaker_plot_handle.setData(x=xvals, y=speaker_channel)
    
        times.append(('neural plots updated', datetime.datetime.now()))

        
        ## Extract onsets and click params
        audio_data_hp, click_params = self.extract_audio_onsets(speaker_channel)
        
        # Drop onsets too close to the edges
        click_params = click_params[
            (click_params['t_samples'] > self.audio_extract_win_samples) &
            (click_params['t_samples'] < 
            len(audio_data_hp) - self.audio_extract_win_samples)
            ]
        
        # Return if no clicks to analyze
        if len(click_params) == 0:
            return
        
        times.append(('clicks extracted', datetime.datetime.now()))
        
        
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

        ## Identify outlier trials
        # Figure out how to handle this in the case that ch2 might be noise
        
        #~ # How much to keep
        #~ quantile = 1

        #~ # Identify trials in the top quantile of abs().max() and std()
        #~ vals1 = triggered_neural.abs().max(1)
        #~ thresh1 = vals1.quantile(quantile)
        #~ mask1 = vals1 > thresh1

        #~ vals2 = triggered_neural.std(1)
        #~ thresh2 = vals2.quantile(quantile)
        #~ mask2 = vals2 > thresh2

        #~ # Combine the masks
        #~ mask = mask1 | mask2
        #~ outlier_trials = triggered_neural.index[mask]
        
        outlier_trials = []
        

        ## Aggregate
        # Average by condition after dropping outlier trials
        avg_by_condition = triggered_neural.drop(outlier_trials).groupby(
            ['polarity', 'amplitude']).mean()

        # Average out polarity
        avg_abrs = avg_by_condition.groupby('amplitude').mean()

        # Compare polarities to measure speaker artefact
        try:
            avg_arts = avg_by_condition.loc[True] - avg_by_condition.loc[False]
        except KeyError:
            avg_arts = None

        # Average audio by condition
        avg_audio_by_condition = triggered_ad.groupby(['amplitude', 'polarity']).mean()
        times.append(('aggregation done', datetime.datetime.now()))


        ## Plot clicks by amplitude
        # First abslog avg_audio_by_condition for display
        avg_audio_by_condition = np.log10(np.abs(avg_audio_by_condition))
        avg_audio_by_condition[avg_audio_by_condition < 0] = 0
        
        # Reindex by amplitudes that should exist so things line up
        avg_audio_by_condition = avg_audio_by_condition.reindex(
            triggered_ad.index.levels[0], level='amplitude')
        
        # Separate negatives and positives
        neg_clicks = avg_audio_by_condition.xs(False, level='polarity')
        pos_clicks = avg_audio_by_condition.xs(True, level='polarity')

        # Plot negatives
        zobj = zip(self.abr_audio_monitor_neg_handle_l, neg_clicks.values)
        for handle, topl in zobj:
            handle.setData(
                x=avg_audio_by_condition.columns.values, y=topl)

        # Plot positives
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

        
        ## Print debug timing information
        times.append(('done', datetime.datetime.now()))
        times = pandas.DataFrame.from_records(times, columns=['event', 't'])
        times['diff'] = times['t'].diff().dt.total_seconds()
        
        # Store total time taken
        self.update_time_taken = times['diff'].sum()
        
        # More verbose output
        self.times_l.append(times)
        concatted = pandas.concat(
            self.times_l, 
            keys=range(len(self.times_l)), 
            names=['rep'])
        #~ print(concatted)
        meaned = concatted.groupby('event')['diff'].mean()

        print(meaned)
        print(meaned.sum())

class MainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self, update_interval_ms=100, experimenter='mouse'):

        ## Superclass QMainWindow init
        super().__init__()


        ## This sets up self._exception_hook to handle any unexpected errors
        self._set_up_exception_handling()
        
        
        ## Parameters that can be set by user interaction
        self.experimenter = experimenter
        
        
        ## Create objects here that would actually do the work, tfw etc
        self.abr_device = ABR_Device.ABR_Device(
            verbose=True, 
            serial_port='/dev/ttyACM0', 
            serial_baudrate=115200, 
            serial_timeout=0.1,
            abr_data_path='/home/mouse/mnt/cuttlefish/surgery/abr_data',
            data_in_memory_duration_s=60,
            experimenter=self.experimenter,
            )        
        
        # Keep track of whether abr_device is running (to avoid multiple
        # clicks on the start button)
        self.experiment_running = False


        ## Create a timer to check for any uncaught errors
        self.timer_check_for_errors = PyQt5.QtCore.QTimer(self)
        
        # Any error that happens in abr_device.update will just crash the
        # timer thread, not the main thread. When this happens, 
        # self._exception_hook is called and sets self.exception_occured.
        # self._check_if_error_occurred will then set the background to red.
        self.timer_check_for_errors.timeout.connect(self._check_if_error_occured)
        self.timer_check_for_errors.start(500)

        # Timers for continuous updating
        # Create a PyQt5.QtCore.QTimer object to continuously update the plot         
        self.timer_update = PyQt5.QtCore.QTimer(self) 
        self.timer_update.timeout.connect(self.update)  
        self.update_interval_ms = update_interval_ms


        ## Create a widget to contain everything
        container_widget = PyQt5.QtWidgets.QWidget(self)

        # Set this one as the central widget
        self.setCentralWidget(container_widget)
        
        
        ## Create a layout
        # Vertical layout: OscilloscopeWidget, and then System Control widget
        container_layout = PyQt5.QtWidgets.QVBoxLayout(container_widget)
        

        ## Top in layout: OscilloscopeWidget
        # Initializing OscilloscopeWidget to show the pokes
        self.oscilloscope_widget = OscilloscopeWidget(self.abr_device)

        # Add to layout
        container_layout.addWidget(self.oscilloscope_widget)

        
        ## Second in layout: System Control
        # Horizontal: buttons, and then grid of params
        system_control_layout = PyQt5.QtWidgets.QHBoxLayout(self) 
        container_layout.addLayout(system_control_layout)

        # Create buttons
        self.set_up_replay_button()
        self.set_up_start_button()
        self.set_up_stop_button()
        
        # Creating vertical layout for buttons
        start_stop_layout = PyQt5.QtWidgets.QVBoxLayout()
        start_stop_layout.addWidget(self.replay_button)
        start_stop_layout.addWidget(self.start_button)
        start_stop_layout.addWidget(self.stop_button)        
        
        # Add the buttons to system_control_layout
        system_control_layout.addLayout(start_stop_layout)
        
        # Grid of params
        row_session_boxes = PyQt5.QtWidgets.QGridLayout()
        system_control_layout.addLayout(row_session_boxes)
        
        # Param: experimenter
        self.line_edit_experimenter = PyQt5.QtWidgets.QLineEdit(
            str(self.experimenter))
        row_session_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('experimenter'), 0, 0)
        row_session_boxes.addWidget(self.line_edit_experimenter, 0, 1)

        # Param: session path
        self.label_session_dir = PyQt5.QtWidgets.QLabel('')
        row_session_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('session dir'), 1, 0)
        row_session_boxes.addWidget(self.label_session_dir, 1, 1)

        # Param: data_collected_s
        self.label_data_collected_s = PyQt5.QtWidgets.QLabel('')
        row_session_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('data collected (s)'), 2, 0)
        row_session_boxes.addWidget(self.label_data_collected_s, 2, 1)

        # Param: packets_in_memory
        self.label_packets_in_memory = PyQt5.QtWidgets.QLabel('')
        row_session_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('packets in memory'), 3, 0)
        row_session_boxes.addWidget(self.label_packets_in_memory, 3, 1)


        ## Set the size and title of the main window
        # Title
        self.setWindowTitle('ABR')
        
        # Size in pixels (can be used to modify the size of window)
        self.resize(1200, 800)
        self.move(100, 100)
        
        # Show it
        self.show()

    def set_up_replay_button(self):
        """Create a replay button and connect to self.replay"""
        # Create button
        self.replay_button = PyQt5.QtWidgets.QPushButton("Replay Session")
        
        # Set style
        self.replay_button.setStyleSheet(
            "background-color : green; color: white;") 

        # Start the abr_device and the updates
        self.replay_button.clicked.connect(self.replay)        
    
    def set_up_start_button(self):
        """Create a start button and connect to self.start"""
        # Create button
        self.start_button = PyQt5.QtWidgets.QPushButton("Start Session")
        
        # Set style
        self.start_button.setStyleSheet(
            "background-color : green; color: white;") 

        # Start the abr_device and the updates
        self.start_button.clicked.connect(self.start)

    def set_up_stop_button(self):
        """Create a stop button and connect to self.stop"""
        # Create button
        self.stop_button = PyQt5.QtWidgets.QPushButton("Stop Session")
        
        # Set style
        self.start_button.setStyleSheet(
            "background-color : green; color: white;") 
        
        # Stop the abr_device and the updates
        self.stop_button.clicked.connect(self.stop)
    
    def _set_up_exception_handling(self):
        # This flag is set True when an exception occurs
        self.exception_occurred = False 
        
        # This flag is set True after it occurs, when it is handled
        self.exception_handled = False
        
        # this registers the exception_hook() function as hook
        sys.excepthook = self._exception_hook

    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        
        It is triggered each time an uncaught exception occurs. 
        # https://timlehr.com/2018/01/python-exception-hooks-with-qt-message-box/
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        else:
            # Flag that an error has occured
            self.exception_occurred = True
            
            # Get the exc_info
            exc_info = (exc_type, exc_value, exc_traceback)
            
            # Form a log message
            log_msg = '\n'.join([
                ''.join(traceback.format_tb(exc_traceback)),
                '{0}: {1}'.format(exc_type.__name__, exc_value),
                ])
            
            # Log it
            print(
                "Uncaught exception:\n {0}".format(log_msg))

    def _check_if_error_occured(self):
        """Called every time a abr_device update is called
        
        Checks self.exception_occurred, which is set by self._exception_hook
        after any uncaught exception. If it is True, set background color to
        red. 
        """
        # Only run this block once, after an error occurs and before it is
        # "handled", although handling really just means logging
        if self.exception_occurred and not self.exception_handled:
            # Set background red
            self.setStyleSheet("background-color: red;") 
            
            # Mark as handled
            self.exception_handled = True
            
            # Stop
            self.stop()

    def replay(self):
        """Replay a previous session
        
        This function is called when the replay button is clicked.
        The user will be asked to choose a binary file in a QFileDialog,
        and then this is passed to `start`.
        """
        # Get a folder
        replay_filename = PyQt5.QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a binary file", 
            #"/home/mouse/mnt/cuttlefish/surgery/abr_data", 
            os.path.expanduser('~/mnt/cuttlefish/abr/LVdata/250109/BG/'),
            "Binary Files (*.bin)",
            )[0]
        
        # Call start using that filename
        self.start(replay_filename=replay_filename)

    def start(self, checkable_state=None, replay_filename=None):
        """Start a session
        
        This function is called when start button is clicked, or indirectly
        when the replay button is clicked.
        
        replay_filename : str or None
            If this is a path to an existing binary file, then self.abr_device
            will replay that file. If that file does not work for some reason, 
            an error is raised.
            
            If this is None, then self.abr_device will collect new data.
        
        Workflow
        * The current value of the "experimenter" line edit is used to set
            the experimenter parameter in self.abr_device, which in turn 
            determines where the data is stored.
        * The individual widgets are started:
            self.abr_device.start_session (using replay_filename)
            self.oscilloscope_widget.start
            self.timer_update.start
        """
        ## Warn if already runing
        if self.experiment_running:
            print(
                'error: you clicked start but the experiment '
                'is already running')
            return
        self.experiment_running = True
        

        ## Set self.experimenter based on self.line_edit_experimenter.text()
        # Get the current value of experimenter
        # This line_edit is not queried at any other time, only at the time
        # the session is started, so don't try to use it for other things 
        # because it is not automatically updated
        text = self.line_edit_experimenter.text()

        # TODO: other validation here such as chars that don't work as a 
        # directory name
        if text == '':
            print(f'warning: {text} is invalid experimenter, ignoring')
            return
        
        # Store self.experimenter
        self.experimenter = text       
        
        # Also update in self.abr_device
        self.abr_device.experimenter = self.experimenter
        
        
        ## Start stuff
        # TODO: Handle the case where the abr_device doesn't actually start
        # because it's not ready
        self.abr_device.start_session(replay_filename=replay_filename)

        # Start plot widgets
        # TODO: consider controlling their timers in this object
        self.oscilloscope_widget.start()
        
        # Start updating MainWindow (such as System Control)
        self.timer_update.start(self.update_interval_ms)

    def stop(self):
        """Called when we want to stop everything
        
        Can happen because stop button was clicked, or because an error
        occurred.
        """
        # Stop the ABR device (serial port, etc)
        self.abr_device.stop_session()
        
        # Stop updating the scope widgets
        self.oscilloscope_widget.stop()
        
        # Stop updating the main window
        self.timer_update.stop()
        
        # Set to False so we can start the session again
        self.experiment_running = False

    def update(self):
        # Set data labels
        if self.abr_device.session_dir is not None:
            self.label_session_dir.setText(self.abr_device.session_dir)
        
        if self.abr_device.tsr is not None:
            self.label_data_collected_s.setText('{:.1f}'.format(
                self.abr_device.tsr.n_packets_read * 500 / 16000))

            self.label_packets_in_memory.setText(str(
                len(self.abr_device.tsr.deq_data)))
        
        #~ self.label_data_written_s = str(
            #~ len(self.abr_device.tfw.n_chunks_written) * 500 / 16000)

    def closeEvent(self, event):
        """Executes when the window is closed
        
        Send 'exit' signal to all IP addresses bound to the GUI
        """
        # Stop ABR device
        #self.abr_device.stop_session()
        event.accept()        
