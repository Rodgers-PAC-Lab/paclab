"""MainWindow of the GUI

MainWindow : object that instantiates the GUI
"""

import sys
import traceback
import datetime
import os
import numpy as np
import PyQt5.QtWidgets
import PyQt5.QtCore
from . import ABR_Device
from . import graphics

class MainWindow(PyQt5.QtWidgets.QMainWindow):
    """MainWindow of the GUI
    
    """
    def __init__(self, update_interval_ms=100, experimenter='mouse'):
        """Instantiate a new MainWindow
        
        """

        ## Superclass QMainWindow init
        super().__init__()


        ## This sets up self._exception_hook to handle any unexpected errors
        self._set_up_exception_handling()
        
        
        ## Parameters that can be set by user interaction
        self.experimenter = experimenter
        
        
        ## Create objects here that would actually do the work, tfw etc
        self.abr_device = ABR_Device.ABR_Device(
            verbose=False, 
            serial_port='/dev/ttyACM0', 
            serial_timeout=0.1,
            abr_data_path='/home/mouse/mnt/cuttlefish/surgery/abr_data',
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
        self.oscilloscope_widget = graphics.OscilloscopeWidget(self.abr_device)

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

        # Param: n_late_reads
        self.label_n_late_reads = PyQt5.QtWidgets.QLabel('N/A')
        row_session_boxes.addWidget(
            PyQt5.QtWidgets.QLabel('# late reads'), 4, 0)
        row_session_boxes.addWidget(self.label_n_late_reads, 4, 1)


        ## Set the size and title of the main window
        # Title
        self.setWindowTitle('ABR')
        
        # Size in pixels (can be used to modify the size of window)
        self.resize(1200, 900)
        self.move(10, 10)
        
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
        self.abr_device.start(replay_filename=replay_filename)

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
        if self.abr_device.running:
            # Stop the ABR device (serial port, etc)
            self.abr_device.stop()
            
            # Stop updating the scope widgets
            self.oscilloscope_widget.stop()
            
            # Stop updating the main window
            self.timer_update.stop()
            
            # Set to False so we can start the session again
            self.experiment_running = False
        
        else:
            print(
                'warning: ignoring stop command because session is not running')

    def update(self):
        # Set data labels
        if self.abr_device.session_dir is not None:
            self.label_session_dir.setText(self.abr_device.session_dir)
        
        if self.abr_device.queue_popper is not None:
            # Seconds of data captured
            captured_data_s = (
                self.abr_device.queue_popper.n_packets_read * 500 / 16000)
            captured_data_min = int(np.floor(captured_data_s / 60))
            captured_data_sec = captured_data_s - 60 * captured_data_min
            
            # Convert to MM:SS format (dropping most of the microsceonds)
            self.label_data_collected_s.setText(
                f'{captured_data_min:02}:{captured_data_sec:04.1f}')

            self.label_packets_in_memory.setText(str(
                len(self.abr_device.deq_data)))
            
            # This doesn't work because this variable isn't shared over 
            # the other process
            #~ self.label_n_late_reads.setText(str(
                #~ self.abr_device.serial_reader.late_reads))
            
        if self.abr_device.file_writer is not None:
            self.label_data_written_s = str(
                self.abr_device.file_writer.n_chunks_written * 500 / 16000)

    def closeEvent(self, event):
        """Executes when the window is closed
        
        Send 'exit' signal to all IP addresses bound to the GUI
        """
        # Stop ABR device
        if self.abr_device.running:
            print('warning: GUI closed while session was running')
            self.abr_device.stop()
        
        # Not sure what this does
        event.accept()        
