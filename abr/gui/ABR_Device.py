"""Objects to run an ABR session

This module contains no GUI code. Everything should also run in a CLI.

The only defined class is ABR_Device, which can run a recording session.
In CLI mode, directly instantiate ABR_Device.
In GUI mode, the MainWindow instantiates and owns it.
"""

import multiprocessing
import time
import datetime
import os
from . import serial_io
from . import file_io

class ABR_Device(object):
    """Runs an ABR session
    
    Attributes
    ---
    serial_reader : serial_io.SerialReader  
        Reads the data
    file_writer : file_io.FileWriter
        Writes the data
    """
    def __init__(self, 
        verbose=True, 
        serial_port='/dev/ttyACM0', 
        serial_timeout=0.1,
        abr_data_path='/home/mouse/mnt/cuttlefish/surgery/abr_data',
        experimenter='mouse',
        ):
        """Initialize a new ABR_Device object to collect ABR data.
        
        Arguments
        ---
        verbose : bool
            If True, write more debugging information
        serial_port : str
            Path to serial port
        serial_timeout : numeric
            Time to wait for a message from the serial port before returning
        abr_data_path : str
            Path to root directory to store data in
        """
        ## Store parameters
        # Currently not supported to change the sampling rate or gains
        self.sampling_rate = 16000
        self.gains = [24, 1, 24, 1, 24, 1, 1, 1] # must be list to be json

        # Control verbosity
        self.verbose = verbose
        
        # Parameters to send to serial port
        self.serial_port = serial_port
        self.serial_timeout = serial_timeout
        
        # Where to save data
        self.abr_data_path = abr_data_path
        self.experimenter = experimenter
        self.session_dir = None # until set
        
        
        ## Instance variables
        # These are None until set
        self.serial_reader = None
        self.file_writer = None
        self.queue_popper = None
        
        # Keep track of whether we're running
        self.running = False
    
    def run_session(self):
        """Called in CLI mode to run a session
        
        Calls self.start(), waits until CTRL+C, calls self.stop()
        """
        self.start()
        
        try:
            while True:
                time.sleep(.1)
                
                #~ if datetime.datetime.now() > (
                    #~ dt_start + datetime.timedelta(seconds=6)):
                    #~ print('reached 6 second shutdown')
                    #~ break
        
        except KeyboardInterrupt:
            print('received CTRL+C, shutting down')
        
        finally:
            self.stop_session()
    
    def determine_session_directory(self):
        """Determine the directory to keep files for a new session in
        
        This will be something like os.path.join(
            abr_data_path, date_string, experimenter, session_number_s)
        
        where abr_data_path is fixed, date_string is like '2025-01-01', 
        experimenter is mandatory and provided, and session_number_s 
        is a string like '0001' that begins at 1 and increments with 
        every new session. 
        
        Within that directory, we will put the individual data files for 
        that session.
        
        This function goes to that directory and finds the lowest session
        number that doesn't yet exist. 
        
        Returns: session_number, session_path
            session_number: int
            session_path: full path to session directory, which does not yet
                exist
        """
        # Date first
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")

        # Path to files for this date
        date_dir = os.path.join(self.abr_data_path, date_string)
        
        # Ensure this is a usable directory
        if os.path.exists(date_dir):
            # Assert it's a directory
            if not os.path.isdir(date_dir):
                raise IOError(f'{date_dir} is not a directory')
        else:
            # Create it
            os.mkdir(date_dir)
            
        # Path to files for this experimenter
        experimenter_dir = os.path.join(date_dir, self.experimenter)
        
        # Ensure this is a usable directory
        if os.path.exists(experimenter_dir):
            # Assert it's a directory
            if not os.path.isdir(experimenter_dir):
                raise IOError(f'{experimenter_dir} is not a directory')
        else:
            # Create it
            os.mkdir(experimenter_dir)
        
        # Existing session names (ie, directories) in experimenter_dir
        existing_session_names = [
            filename for filename in os.listdir(experimenter_dir) 
            if os.path.isdir(os.path.join(experimenter_dir, filename))]
        
        # Find the lowest session number that doesn't exist in experimenter_dir
        session_number = 1
        while True:
            # Form a session number string: 001, 002, etc
            session_number_str = '{:03d}'.format(session_number)
            
            # See if any experimenter_file starts with that session_number
            match_found = False
            for existing_session_name in existing_session_names:
                if existing_session_name.startswith(session_number_str):
                    match_found = True
                    break
            
            if match_found:
                # A match was found, increment and try again
                session_number += 1
                
                # Prevents infinite loop
                if session_number >= 999:
                    raise ValueError('unable to determine session number')
            else:
                # No match found, this is the current session number
                break   
        
        # Form session_dir
        session_dir = os.path.join(experimenter_dir, session_number_str)
        assert not os.path.exists(session_dir)
        
        return session_number, session_dir
        
    def start(self, replay_filename=None):
        """Start an ABR session
        
        Arguments
        ---
        replay_filename : path
            If this is not None, replay data from this path instead of
            reading from serial port.
        """
        
        ## Log
        print('starting abr session...')
        if self.verbose:
            print(f'replay filename is {replay_filename}')        
    
        # Store the datetime str for the current session
        self.session_dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Keep track of whether we're running
        self.running = True
    
       
        ## Depends on if we're in live mode
        if replay_filename is None:
            
            ## Define a session_name and create the session_directory
            self.session_number, self.session_dir = (
                self.determine_session_directory())
            
            if self.verbose:
                print(f'creating output directory {self.session_dir}')
            os.mkdir(self.session_dir)
            
            
            ## Create the serial_reader object
            self.serial_reader = serial_io.SerialReader(
                serial_port=self.serial_port,
                serial_timeout=self.serial_timeout,
                gains=self.gains,
                sampling_rate=self.sampling_rate,
                verbose=self.verbose,
                session_dir=self.session_dir,
                session_dt_str=self.session_dt_str,
                )
            
            
            ## Start acquistion in a separate Process
            self.proc = multiprocessing.Process(target=self.serial_reader.start)
            if self.verbose:
                print('starting process')
            self.proc.start()
            
            
            ## Create QueuePopper to pop data into deques
            # Continuously read data out of the mp queues and into 
            # thread-safe deques
            self.queue_popper = serial_io.QueuePopper(
                q_data=self.serial_reader.q_data,
                q_headers=self.serial_reader.q_header,
                verbose=self.verbose,
                )
            
            # Get the default output deqs, which are for gui
            self.deq_header = self.queue_popper.deq_header
            self.deq_data = self.queue_popper.deq_data
            
            # Get outputs for threaded file writer
            # (Must do this before starting queue popper)
            fw_deq_header, fw_deq_data = self.queue_popper.get_output_deqs()
            
            # Start the queue popper
            self.queue_popper.start()
            
            
            ## Continuously write data from the tfw_deqs to disk
            output_filename = os.path.join(
                self.session_dir, 'data.bin')
            output_header_filename = os.path.join(
                self.session_dir, 'packet_headers.csv')
            
            # Create file_writer
            self.file_writer = file_io.FileWriter(
                deq_data=fw_deq_data,
                deq_header=fw_deq_header,
                output_filename=output_filename,
                output_header_filename=output_header_filename,
                )
            
            # Start writing
            self.file_writer.start()
        
        else:
            # Implement replay mode here
            pass
        
        if self.verbose:
            print('done with ABR_device.start')

    def stop(self):
        """Stop experiment
        
        Stops serial_reader
        Stops queue_popper
        
        
        """
        if self.verbose:
            print('ABR_device stopping abr session...')
        
        # Tell the serial_reader to stop reading
        if self.verbose:
            print('setting stop event')
        self.serial_reader.stop_event.set()
        
        # Wait for it to complete
        # Might take a little while because it has to send the stop message, etc
        time.sleep(1)
        
        # Tell the queue_popper to stop
        # This joins, so it blocks until it finishes
        if self.verbose:
            print('stopping queue_popper')
        self.queue_popper.stop()
        
        # Join on the serial_reader (this will hang until all data is read out)
        if self.verbose:
            print('joining')
        self.proc.join(timeout=1)
        
        # If it didn't finish (most likely because data is left in the queues
        # for some reason) then kill it
        if self.proc.is_alive():
            print('warning: could not join serial_reader process; killing')
            self.proc.terminate()

        # Tell the writer to stop
        # This joins, so it blocks until it finishes
        self.file_writer.stop()
        
        # Keep track of whether we're running
        self.running = False
        
        print('ABR_Device shutdown complete')    

