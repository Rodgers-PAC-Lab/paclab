"""Objects to run an ABR session

No GUI code here!
"""

import collections
import threading
import time
import datetime
import os
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
        data_in_memory_duration_s=60,
        experimenter='mouse',
        ):
        """Initialize a new ABR_Device object to collect ABR data.
        
        verbose : bool
            If True, write more debugging information
        
        serial_port : str
            Path to serial port
        
        serial_baudrate : numeric
            Baudrate to use
        
        serial_timeout : numeric
            Time to wait for a message from the serial port before returning
        
        abr_data_path : str
            Path to root directory to store data in
        
        data_in_memory_duration_s : numeric
            No data will be removed from the deque in memory and written to
            disk until the length of the deque exceeds this amount. This is
            also the maximum amount of data that can be analyzed by the GUI.
            Increasing this value gives a more accurate representation of the
            ABR, but it will slow GUI updates, and delay writing to disk. 
        """
        ## Store parameters
        # Currently not supported to change the sampling rate or gains
        self.sampling_rate = 16000
        self.gains = [24, 1, 24, 1, 1, 1, 1, 1] # must be list to be json

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
        
        # How much data to keep in memory
        self.data_in_memory_duration_s = data_in_memory_duration_s
        
        
        ## Instance variables
        # These are None until set
        self.ser = None
        self.tsr = None
        self.tfw = None
    
    def run_session(self):
        self.start_session()
        
        try:
            while True:
                time.sleep(.1)
        
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
            ## Open a serial connection to the teensy
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                timeout=self.serial_timeout,
                )
            
            
            ## Close out the previous session
            print('flushing serial port')
            
            # Flush
            self.ser.flushInput()
            self.ser.flush()
            
            # Stop data acquisition if it's already happening
            serial_comm.write_stop(
                self.ser, warn_if_not_running=False, warn_if_running=True)
            
            
            ## Query the serial port
            # Just a consistency check because the answer should always be the same
            query_res = serial_comm.write_query(self.ser)


            ## Set parameters
            # Form the command
            # TODO: document these other parameters
            str_gains = ','.join([str(gain) for gain in self.gains])
            cmd_str = f'U,8,{self.sampling_rate},0,{str_gains};'
            
            # Log
            print(f'setting parameters: {cmd_str}')
            
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


            ## Tell it to start
            print('starting acquisition...')
            serial_comm.write_start(self.ser)

            
            ## Start acquiring
            self.tsr = ThreadedSerialReader(self.ser, verbose=False)
            self.tsr.start()


            ## Start the TFW
            # This parameter determines how much data is kept in memory before
            # writing to disk
            # We want to make sure at least that many chunks are kept in the deque
            # before writing out
            minimum_deq_length = int(np.rint(
                self.data_in_memory_duration_s * self.sampling_rate / 500))
            
            # Create tfw
            self.tfw = ThreadedFileWriter(
                self.tsr.deq_data, 
                self.tsr.deq_headers,
                verbose=False,
                output_filename=os.path.join(self.session_dir, 'data.bin'),
                output_header_filename=os.path.join(self.session_dir, 'packet_headers.csv'),
                minimum_deq_length=minimum_deq_length,
                )
            self.tfw.start()
        
        else:
            ## This is the canned mode
            # In this case the deque will never be emptied
            # TODO: initialize a dummy ThreadedFileWriter to do nothing
            # but empty the deque
            self.tsr = ThreadedFileReader(replay_filename)
            self.tsr.start()


            ## Start the TFW
            # This parameter determines how much data is kept in memory before
            # writing to disk
            # We want to make sure at least that many chunks are kept in the deque
            # before writing out
            minimum_deq_length = int(np.rint(
                self.data_in_memory_duration_s * self.sampling_rate / 500))
            
            # Create tfw
            self.tfw = ThreadedFileWriter(
                self.tsr.deq_data, 
                self.tsr.deq_headers,
                verbose=False,
                output_filename=None,
                output_header_filename=None,
                minimum_deq_length=minimum_deq_length,
                )
            self.tfw.start()            

    def stop_session(self):
        print('stopping abr session...')
        # Tell the thread to stop reading - it will keep reading until the 
        # serial buffer is empty though
        # I think because of the `join`, this will block until the serial
        # buffer is empty
        if self.tsr is not None:
            self.tsr.stop()
        
        # Wait for the file writer finish - this just empties the deq, and
        # is insensitive to serial traffic
        if self.tfw is not None:
            self.tfw.stop()
        
        # These steps can only be taken if the serial port was created, which
        # doesn't happen for errors that occur on startup
        if self.ser is not None:
            # Tell the device to stop producing data
            serial_comm.write_stop(self.ser)
        
            # Close serial port
            self.ser.close()
        else:
            print('error: cannot close serial port because it was never created')
        
        print("abr device closed")

class ThreadedFileWriter(object):
    def __init__(self, 
        deq_data, deq_headers, output_filename, output_header_filename,
        minimum_deq_length=1000, verbose=False):
        """Initialize a new ThreadedFileWriter

        Pops data from the left side of deq_data and writes to disk.

        deq_data : deque 
            Data filled by and shared with ThreadedSerialReader
            These are the chunks of data, with time along the rows
        
        deq_headers : deque 
            Data filled by and shared with ThreadedSerialReader
            These are the headers for each chunk of data, one row per chunk

        output_filename : str or None
            Where to write out data
            if None, nothing is written to disk
        
        output_header_filename : str or None
            Where to write out headers
            if None, nothing is written to disk

        minimum_deq_length : int
            If len(deq) < minimum_deq_length, no data will be popped or written
            This ensure there is always recent data to visualize
        """
        # Store
        self.output_filename = output_filename
        self.output_header_filename = output_header_filename
        self.minimum_deq_length = minimum_deq_length
        self.deq_data = deq_data
        self.deq_headers = deq_headers
        self.keep_writing = True
        self.thread = None
        self.verbose = verbose
        self.n_chunks_written = 0
        
        # Set to null by default
        if self.output_filename is None:
            self.output_filename = os.devnull
        
        if self.output_header_filename is None:
            self.output_header_filename = os.devnull
        
        #~ # Keep track of big_data here
        #~ self.big_data_last_col = 0
        #~ self.big_data = None
        #~ self.headers_l = []
        
        # TODO: make this match the rest of the code
        self.header_colnames = [
            'header_size',
            'header_nbytes',
            'data_format_enum',
            'data_format',
            'total_packets',
            'packet_num',
            'data_class',
            'n_samples',
            ]
        
        # Erase the file
        with open(self.output_filename, 'wb') as output_file:
            pass
        
        # Write the headers
        with open(self.output_header_filename, 'a') as headers_out:
            str_to_write = ','.join(self.header_colnames)
            headers_out.write(str_to_write + '\n')            

    def write_to_disk(self, drain=False):
        # Don't write if the deq is too short, unless drain is True
        if drain:
            threshold = 0
        else:
            threshold = self.minimum_deq_length
        
        # Empty 
        with (
                open(self.output_filename, 'ab') as data_out, 
                open(self.output_header_filename, 'a') as headers_out
                ):
            while len(self.deq_data) > threshold:
                ## Pop
                # Pop the oldest data
                data_chunk = self.deq_data.popleft()
                
                # Pop the oldest header
                data_header = self.deq_headers.popleft()
            
                
                ## Append raw data to output file
                # Note: just maintains dtype of whatever data_chunk is
                data_out.write(data_chunk)
                
                # Append header
                str_to_write = ','.join(
                    [str(data_header[colname]) for colname in self.header_colnames])
                headers_out.write(str_to_write + '\n')
                

                #~ ## Append to big data
                #~ if self.big_data is None:
                    #~ # Special case, it doesn't exist yet
                    #~ # This is also how we find out how many columns it has
                    #~ self.big_data = data_chunk.copy()
                    #~ self.big_data_last_col = len(data_chunk)
                
                #~ else:
                    #~ # The normal case, self.big_data does exist
                    #~ # This is how long it will be
                    #~ self.new_big_data_last_col = (
                        #~ self.big_data_last_col + len(data_chunk))

                    #~ # Grow if needed
                    #~ if self.new_big_data_last_col > len(self.big_data):
                        #~ # Make it twice as big as needed
                        #~ new_len = 2 * self.new_big_data_last_col
                        
                        #~ # Fill it with zeros
                        #~ self.new_big_data = np.zeros(
                            #~ (new_len, self.big_data.shape[1]))
                        
                        #~ # Copy in the old data
                        #~ self.new_big_data[:self.big_data_last_col] = (
                            #~ self.big_data[:self.big_data_last_col])
                        
                        #~ # Rename
                        #~ self.big_data = self.new_big_data
                    
                    #~ # Add the new data at the end
                    #~ self.big_data[
                        #~ self.big_data_last_col:self.new_big_data_last_col] = (
                        #~ data_chunk)

                    #~ # Update the pointer
                    #~ self.big_data_last_col = self.new_big_data_last_col

                #~ # Store read headers
                #~ self.headers_l.append(data_header)

                
                ## Keep track of how many chunks written
                self.n_chunks_written += 1
        
        if self.verbose:
            print(f"popped and wrote {self.n_chunks_written} chunks")

    def write_out(self):
        """Target of the thread
        
        As long as self.keep_reading is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        Keep track of any late reads.
        
        Once self.keep_reading is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_writing
        while self.keep_writing:
            self.write_to_disk()
            
            # This sleeps keeps us from writing too frequently, which is 
            # probably somewhat expensive
            # But it increases the latency before draining
            time.sleep(.3)
        
        # self.keep_reading has been set False
        # Read any last chunks
        self.write_to_disk(drain=True)
    
    def start(self):
        """Start the capture of data"""
        self.thread = threading.Thread(target=self.write_out)
        self.thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_writing = False
        
        if self.thread is not None:
            self.thread.join()

class ThreadedFileReader(object):
    """Instantiates a thread to constantly read from a previous file
    
    This is for debugging, so we can load data stored by Bill's code (not
    the same format as our code!) and see what it would look like.
    
    self.deq : collections.deque
        Data will be appended to the right side as it comes in
        The oldest data will be on the left

    self.thread : the thread that reads
    """
    def __init__(self, filename, verbose=False):
        """Instatiate a new ThreadedFileReader
        
        filename : full path to a *.bin written by BG code
        
        """
        self.deq_headers = collections.deque()
        self.deq_data = collections.deque()
        self.filename = filename
        self.keep_reading = True
        self.late_reads = 0
        self.n_packets_read = 0
        self.thread = None
        self.verbose = verbose
        
        
        ## Read the entire file
        # Read the data
        with open(filename, "rb") as fi:
            data_bytes = fi.read()

        # We need to parse the first header separately because it has data that
        # we need to parse the rest of the packets.
        first_header_bytes = data_bytes[:60]
        self.first_header_info = paclab.abr.labview_loading.parse_header(
            first_header_bytes)

        # Make it match expected format
        self.first_header_info['header_size'] = 0
        self.first_header_info['header_nbytes'] = 0
        self.first_header_info['data_format_enum'] = 0
        self.first_header_info['data_format'] = 0
        self.first_header_info['total_packets'] = 0
        self.first_header_info['data_class'] = 0
        self.first_header_info['n_samples'] = 0

        # Parse the entire file
        self.data = paclab.abr.labview_loading.parse_data(
            data_bytes,
            self.first_header_info['number_channels'],
            self.first_header_info['number_samples'])
        
        # Invert the processing done by BG to restore it to bytes
        # It's stored in V
        # Re-apply the gain applied by ADS1299
        self.data = self.data * np.array([24, 1, 24, 1, 1, 1, 1, 1])
        
        # Convert from V to bits (full-scale range is 9V)
        self.data = self.data * 2 ** 24 / 9
        
        # Convert to bytes in C-order (needed for writing)
        self.data = self.data.astype(np.int32, order='C')
       
        
        ## Get the start time and parcel out data accordingly
        self.start_time = datetime.datetime.now()
    
    def read_and_append(self):
        # Wait until it's been long enough
        inter_packet_interval_s = 500 / 16000
        release_time = self.start_time + datetime.timedelta(
            seconds=inter_packet_interval_s * self.n_packets_read)
        
        while datetime.datetime.now() < release_time:
            time.sleep(inter_packet_interval_s / 3)
        
        # Get the packet
        payload = self.data[
            self.n_packets_read * 500:(self.n_packets_read + 1) * 500,
            :]
        
        # Append to the left side of the deque
        header = self.first_header_info.copy()
        header['packet_num'] = self.n_packets_read
        self.deq_headers.append(header)
        self.deq_data.append(payload)
        
        # Log
        self.n_packets_read += 1

    def capture(self):
        """Target of the thread
        
        As long as self.keep_reading is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        Keep track of any late reads.
        
        Once self.keep_reading is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_reading
        while self.keep_reading:
            if self.verbose:
                print(f'deqlen: {len(self.deq_data)}')
            self.read_and_append()
    
    def start(self):
        """Start the capture of data"""
        self.thread = threading.Thread(target=self.capture)
        self.thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_reading = False
        
        if self.thread is not None:
            self.thread.join()

class ThreadedSerialReader(object):
    """Instantiates a thread to constantly read from serial into a deque
    
    self.ser : serial.Serial, provided by user

    self.deq : collections.deque
        Data will be appended to the right side as it comes in
        The oldest data will be on the left

    self.late_reads : int
        The number of times that more than one packet of data was
        available. This should ideally never happen, because the buffer
        overflows around 65000 bytes or so, which is just a few packets
    
    self.thread : the thread that reads
    """
    def __init__(self, ser, verbose=False):
        """Instatiate a new ThreadedSerialReader
        
        ser : serial.Serial
            Data will be read from this object and stored in self.deq
        """
        self.deq_headers = collections.deque()
        self.deq_data = collections.deque()
        self.ser = ser
        self.keep_reading = True
        self.late_reads = 0
        self.n_packets_read = 0
        self.thread = None
        self.verbose = verbose
    
    def read_and_append(self):
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
            
        # Store the time
        if self.verbose:
            print(f"read packet {data_packet['message']['packet_num']}")
    
        # Append to the left side of the deque
        self.deq_headers.append(data_packet['message'])
        self.deq_data.append(data_packet['payload'])
        
        # Log
        self.n_packets_read += 1

    def capture(self):
        """Target of the thread
        
        As long as self.keep_reading is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        Keep track of any late reads.
        
        Once self.keep_reading is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_reading
        while self.keep_reading:
            if self.verbose:
                print(f'deqlen: {len(self.deq_data)}')
            self.read_and_append()
        
        # self.keep_reading has been set False
        # Read any last chunks
        while self.ser.in_waiting > 0:
            # Probably should only do this once, because if it's more than
            # one, something has gone wrong 
            self.read_and_append()
    
    def start(self):
        """Start the capture of data"""
        self.thread = threading.Thread(target=self.capture)
        self.thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_reading = False
        
        if self.thread is not None:
            self.thread.join()


