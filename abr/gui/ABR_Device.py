"""Objects to run an ABR session

No GUI code here!
"""

import collections
import threading
import multiprocessing
import time
import datetime
import os
import signal
import json
import serial
from . import serial_comm
import numpy as np
import paclab.abr

class ABR_Device(object):
    def __init__(self, 
        verbose=True, 
        serial_port='/dev/ttyACM0', 
        serial_baudrate=115200, 
        serial_timeout=0.1,
        abr_data_path='/home/mouse/mnt/cuttlefish/surgery/abr_data',
        experimenter='mouse',
        ):
        """Initialize a new ABR_Device object to collect ABR data.
        
        verbose : bool
            If True, write more debugging information
        
        serial_port : str
            Path to serial port
        
        serial_baudrate : numeric
            Baudrate to use
            I suspect this is ignored. The actual data transfer rate is
            16KHz * 8ch * 4B = 512KB/s, well over this baudrate.
        
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
        self.serial_baudrate = serial_baudrate
        self.serial_timeout = serial_timeout
        
        # Where to save data
        self.abr_data_path = abr_data_path
        self.experimenter = experimenter
        self.session_dir = None # until set
        
        
        ## Instance variables
        # These are None until set
        self.ser = None
        self.tsr = None
        self.tfw = None
    
    def run_session(self):
        dt_start = datetime.datetime.now()
        self.start_session()
        
        try:
            while True:
                time.sleep(.1)
                
                #~ if datetime.datetime.now() > dt_start + datetime.timedelta(seconds=6):
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
        
    def start_session(self, replay_filename=None):
        print('starting abr session...')
        print(f'replay filename is {replay_filename}')        
        

        ## Store the datetime str for the current session
        self.session_dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


        ## Define a session_name and create the session_directory
        # Only create the output directory if this is a live session
        if replay_filename is None:
            self.session_number, self.session_dir = (
                self.determine_session_directory())
            
            print(f'creating output directory {self.session_dir}')
            os.mkdir(self.session_dir)
        
        
        ## These steps only occur if we're in live mode
        if replay_filename is None:
            # Create the serial_reader object
            self.serial_reader = MultiprocSerialReader(
                serial_port=self.serial_port,
                serial_timeout=self.serial_timeout,
                gains=self.gains,
                sampling_rate=self.sampling_rate,
                verbose=False,
                session_dir=self.session_dir,
                session_dt_str=self.session_dt_str,
                )
            
            # Start it
            self.proc = multiprocessing.Process(target=self.serial_reader.start_session)
            print('starting')
            self.proc.start()
            
            # Continuously read data out of the mp queues and into a thread-safe
            # deque
            self.tqr = ThreadedQueueReader(
                q_data=self.serial_reader.output_data,
                q_headers=self.serial_reader.output_headers,
                )
            
            # Get the default output deqs, which are for gui
            self.deq_header = self.tqr.deq_header
            self.deq_data = self.tqr.deq_data
            
            # Get outputs for threaded file writer
            tfw_deq_header, tfw_deq_data = self.tqr.get_output_deqs()
            
            # Start
            self.tqr.start()
            
            # Continuously write data from the tfw_deqs to disk
            self.tfw = ThreadedFileWriter(
                deq_data=tfw_deq_data,
                deq_headers=tfw_deq_header,
                output_filename=os.path.join(self.session_dir, 'data.bin'),
                output_header_filename=os.path.join(self.session_dir, 'packet_headers.csv'),
                )
            
            self.tfw.start()
        
        print('done with ABR_device.start_session')

    def stop_session(self):
        print('ABR_device stopping abr session...')
        # Tell the serial_reader to stop reading
        print('setting stop event')
        self.serial_reader.stop_event.set()
        
        # Wait for it to complete
        # Might take a little while because it has to send the stop message, etc
        time.sleep(1)
        
        # Tell the threaded_queue_reader to stop
        print('stopping tqr')
        self.tqr.keep_reading = False
        
        # Wait for it to complete
        # This one should be fast
        time.sleep(0.1)
        
        # Join on the serial_reader (this will hang until all data is read out)
        print('joining')
        self.proc.join(timeout=1)
        if self.proc.is_alive():
            print('killing')
            self.proc.terminate()

        # Tell the writer to stop
        self.tfw.stop()
        
        print('done')    

