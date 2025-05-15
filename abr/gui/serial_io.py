"""Objects for reading from the serial port

SerialReader - reads data from the ADS1299 in a separate process
QueuePopper - pops data from SerialReader's queues into thread-safe deques
"""
import collections
import threading
import datetime
import multiprocessing
import time
import os
import signal
import json
import serial
from . import serial_comm
import numpy as np

class SerialReader(object):
    """Object that reads from the serial port in another process
    
    This object has to run in its own process because otherwise the GUI
    (or GIL?) can delay data reading for so long that it is lost from the 
    serial port. 
    
    SerialReader handles all communication on the serial port because 
    serial.Serial objects cannot be pickled across processes.
    
    All output from this object goes through multiprocessing.Queues
    
    Attributes
    ---
    q_data : multiprocessing.Queue()
        Data packets (blocks of data as a 2d numpy array) are pushed to 
        this queue as they arrive
    q_header : multiprocessing.Queue()
        Metadata (dict) are pushed to this queue as they arrive
        One header is pushed for every packet
    stop_event : multiprocessing.Event()
        When this event is set, acquisition stops
    n_packets_read : int
        Incremented every time a packet is read
    late_reads : int
        Incremented every time data is left in the serial buffer after reading
    
    Methods
    ---
    start : Called to start acquisition
        Generally this is the target of a multiprocessing call in 
        ABR_Device
    _read_and_append : Get a packet and push onto the output queues
    """
    def __init__(self, serial_port, serial_timeout, gains, sampling_rate, 
        session_dir, session_dt_str, verbose=True):
        """Initialize a new SerialReader to read from the serial port
        
        serial_port : str, like '/dev/ttyACM0'
        serial_timeout : numeric
            Used to initialize self.ser, a serial.Serial
            This is the number of seconds it will wait for data
        gains : list-like of length 8
            The gain of each channel, sent to ADS1299
        sampling_rate : numeric
            Sampling rate to send to ADS1299
        session_dir : path
            The file `config.json` will be written here
        session_dt_str : str
            Store as key 'session_start_time' in config.json
        verbose : bool
            If True, writes out more information than otherwise
        """
        # Store params
        self.serial_port = serial_port
        self.serial_timeout = serial_timeout
        self.gains = gains
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.session_dir = session_dir
        self.session_dt_str = session_dt_str
        
        # Output goes here
        self.q_data = multiprocessing.Queue()
        self.q_header = multiprocessing.Queue()
        
        # Diagnostics
        self.late_reads = 0
        self.n_packets_read = 0
        
        # This is the flag used for stopping
        self.stop_event = multiprocessing.Event()
    
    def start(self):
        """Acquire data from the ADS1299 until self.stop_event is set
        
        This should be run as the target of a multiprocessing call.
        
        Workflow
        ---
        * Disables CTRL+C in this process
        * Creates the serial port
        * Flush the serial port
        * Write STOP to ADS1299
        * QUERY the ADS1299 
        * Set parameters of ADS1299
        * Writes config.json (containing query results, and a few other things)
          to the session_dir
        * Start data acquisition
        * Continuously call self._read_and_append() until stop_event is set
        * Write STOP to ADS1299
        * Close the serial port
        """
        ## Have the child process ignore CTRL+C
        # Otherwise CTRL+C as a message to stop the main proc also stops this
        # one in an awkard place
        # A side effect of this may be to make it harder to kill this process
        # May want to comment this out for full GUI operation
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        
        ## Create port
        self.ser = serial.Serial(
            port=self.serial_port,
            timeout=self.serial_timeout,
            )
        
        
        ## Close out the previous session
        if self.verbose:
            print(f'{datetime.datetime.now()}: flushing serial port')
        
        # Flush
        self.ser.flushInput()
        self.ser.flush()
        
        # Stop data acquisition if it's already happening
        if self.verbose:
            print('writing stop')
        serial_comm.write_stop(
            self.ser, warn_if_not_running=False, warn_if_running=True)
        
        
        ## Query the serial port
        # Just a consistency check because the answer should always be the same
        if self.verbose:
            print(f'{datetime.datetime.now()}: querying')
        query_res = serial_comm.write_query(self.ser)


        ## Set parameters
        # Form the command
        # TODO: document these other parameters
        str_gains = ','.join([str(gain) for gain in self.gains])
        cmd_str = f'U,8,{self.sampling_rate},0,{str_gains};'
        
        # Log
        if self.verbose:
            print(f'{datetime.datetime.now()}: setting parameters: {cmd_str}')
        
        # Log the configs that were used
        to_json = query_res['message']
        
        # Store configs from this object
        to_json['gains'] = self.gains
        to_json['sampling_rate'] = self.sampling_rate
        to_json['session_start_time'] = self.session_dt_str
        
        # Mistakenly, this was int64 in early versions of this software
        # There is currently no support for changing the output_dtype
        # This makes the output_dtype explicit, and also works as a flag
        # for when this fix went into effect
        to_json['output_dtype'] = 'int32'
        
        # Write the config file
        with open(os.path.join(self.session_dir, 'config.json'), 'w') as fi:
            json.dump(to_json, fi)
            
        # Send the command
        serial_comm.write_setupU(self.ser, cmd_str)


        ## Start acquisition and continue until stop event
        if self.verbose:
            print(f'{datetime.datetime.now()}: starting acquisition...')
        serial_comm.write_start(self.ser)
        
        # Capture until we are told to stop
        while not self.stop_event.is_set():
            self._read_and_append()
        
        
        ## Stop event was set, so stop
        if self.verbose:
            print(f'{datetime.datetime.now()}: stop event detected')

        # Tell the device to stop producing data
        serial_comm.write_stop(self.ser)
    
        # Close serial port
        self.ser.close()

        if self.verbose:
            print(
                f'{datetime.datetime.now()}: '
                'done with SerialReader.start')

    def _read_and_append(self):
        """Read a data packet and apend to the output queues
        
        Workflow
        * Read a data packet using serial_comm.read_and_classify_packet
          Raise an exception if no packet is read
        * Add 'late_read' and 'in_waiting' to the header, to log when data
          is leftover in the serial port
        * put_nowait the header and data into self.q_header and
          self.q_data
        * Increment self.n_packets_read
        """
        # Read a data packet
        # Relatively long wait_time here, in hopes that if something is
        # screwed up we can find sync bytes and get back on track
        data_packet = serial_comm.read_and_classify_packet(
            self.ser, wait_time=0.5, assert_packet_type='data')
        
        # If any data remains, this was a late read
        in_waiting = self.ser.in_waiting
        if in_waiting > 0:
            print(f'late read! {in_waiting} bytes in waiting')
            self.late_reads += 1  
            data_packet['message']['late_read'] = True
            data_packet['message']['in_waiting'] = in_waiting
        else:
            data_packet['message']['late_read'] = False
            data_packet['message']['in_waiting'] = in_waiting

        # This is very verbose
        #~ if self.verbose:
            #~ print(f"read packet {data_packet['message']['packet_num']}")
    
        # put_nowait means put right away, else raise an exception
        self.q_header.put_nowait(data_packet['message'])
        self.q_data.put_nowait(data_packet['payload'])
        
        # Log
        self.n_packets_read += 1

class QueuePopper(object):
    """Continuously read from multiprocessing.Queues into deques
    
    This object's job is to pop data from SerialReader's multiprocessing.Queues
    and put them into thread-safe deques. A separate deque is used for 
    each downstream sink. 
    
    It's okay for this to be in a thread because it's okay if it's a little
    delayed from time to time, and its only outputs are thread-safe.
    
    Attributes
    ---
    self.q_data and self.q_header : multiprocessing Queue
        Should be SerialReader.q_data and SerialReader.q_header
        Data is popped from these queues
    self.n_packets_read : int
        How many packets have been popped
    self.deq_data_l, self.deq_header_l : lists of collections.Deque
        Data popped from the queues is pushed into each of these deques
    self.deq_header : collections.Deque, and first entry in self.deq_header_l
        This deque is designated for the GUI to use
    self.deq_data : collections.Deque, and the first entry in self.deq_data_l
        This deque is designated for the GUI to use
    
    Methods
    ---
    get_output_deq : Used to request a new pair of deq_header and deq_data
        for a new sink.
    start : Start popping data (in a thread called self._capture_thread)
    stop : Stop popping data and join self._capture_thread
    """
    def __init__(self, q_data, q_headers, verbose=False):
        """Instatiate a new QueuePopper
        
        Arguments
        ---
        q_data : multiprocessing.Queue
            Should be SerialReader.q_data
        q_header : multiprocessing.Queue
            Should be SerialReader.q_header
        verbose : bool
            If True, more output will be printed
        """
        # Store provided arguments
        self.q_data = q_data
        self.q_headers = q_headers
        self.verbose = verbose
        
        # Used to tell it stop running
        self.keep_reading = True
        
        # Used to keep track of how many packets read
        self.n_packets_read = 0
        
        # Used to keep track of its thread
        self._capture_thread = None
        
        # This is a list of output deques, one per sink
        self.deq_data_l = []
        self.deq_header_l = []

        # Initialize the first deqs, which are used by the GUI
        # Other deques can be requested later
        self.deq_header, self.deq_data = self.get_output_deqs()
    
    def get_output_deqs(self):
        """Create and return new deques that will be filled with data
        
        Keeps a list of all output sources, including a default one
        
        Returns: new_deq_header, new_deq_data
        """
        # Create
        new_deq_header = collections.deque()
        new_deq_data = collections.deque()
        
        # Append
        self.deq_header_l.append(new_deq_header)
        self.deq_data_l.append(new_deq_data)
        
        # Return
        return new_deq_header, new_deq_data
    
    def _capture(self):
        """Target of the thread
        
        As long as self.keep_reading is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        
        Once self.keep_reading is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_reading 
        # Make sure keep_reading is set False after the upstream q is shut
        # down, otherwise we won't get the last bit of data
        while self.keep_reading:
            
            # TODO: what if we get a header but not a packet, or vice versa?
            
            ## Try to get a header
            try:
                header = self.q_headers.get_nowait()
            except multiprocessing.queues.Empty:
                header = None
            
            if header is not None:
                # Append to left side
                for deq_header in self.deq_header_l:
                    deq_header.append(header)
            
            
            ## Try to get a packet of data
            try:
                data = self.q_data.get_nowait()
            except multiprocessing.queues.Empty:
                data = None
            
            if data is not None:
                # Append to left side
                for deq_data in self.deq_data_l:
                    deq_data.append(data)
                
                # Count
                self.n_packets_read += 1
            
            
            ## If we didn't get any data, sleep for a bit
            if header is None or data is None:
                time.sleep(0.3)
        
        # TODO: get any last chunks here
   
    def start(self):
        """Start the capture of data"""
        self._capture_thread = threading.Thread(target=self._capture)
        self._capture_thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_reading = False
        
        if self._capture_thread is not None:
            self._capture_thread.join()


