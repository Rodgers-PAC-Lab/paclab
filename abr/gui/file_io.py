"""Classes for reading and writing from files

FileWriter : Write data to a file in a thread
FileReader : Read data from a file, simulating the ADS1299
"""
import collections
import threading
import time
import datetime
import os
import numpy as np
import paclab.abr

class FileWriter(object):
    """Writes data from deques to disk
    
    Methods
    ---
    start : start writing data in a thread
    stop : stop writing data and join the thread
    """
    def __init__(self, 
        deq_data, deq_header, output_filename, output_header_filename,
        verbose=False):
        """Initialize a new ThreadedFileWriter

        Pops data from the left side of deq_data and writes to disk.

        deq_data : deque 
            Data filled by and shared with QueuePopper
            These are the chunks of data, with time along the rows
        
        deq_header : deque 
            Data filled by and shared with QueuePopper
            These are the headers for each chunk of data, one row per chunk

        output_filename : str or None
            Where to write out data
            if None, nothing is written to disk
        
        output_header_filename : str or None
            Where to write out headers
            if None, nothing is written to disk

        verbose : bool
            If True, write out more stuff
        """
        # Store arguments
        self.output_filename = output_filename
        self.output_header_filename = output_header_filename
        self.deq_data = deq_data
        self.deq_header = deq_header
        self.verbose = verbose
        
        # Whether to keep going
        self.keep_writing = True
        
        # Keep track of my thread
        self.thread = None
        
        # Keep track of how many chunks written
        self.n_chunks_written = 0
        
        # Set to null by default
        if self.output_filename is None:
            self.output_filename = os.devnull
        
        if self.output_header_filename is None:
            self.output_header_filename = os.devnull
        
        # This will be set by the first header that's received
        self.header_colnames = None
        
        # Erase the file
        with open(self.output_filename, 'wb') as output_file:
            pass
        
        # Write the headers
        with open(self.output_header_filename, 'w') as headers_out:
            pass

    def write_to_disk(self):
        """Write data to disk"""
        # Return immediately if nothing to do
        if len(self.deq_data) == 0:
            return
        
        # Otherwise write data until we run out
        with (
                open(self.output_filename, 'ab') as data_out, 
                open(self.output_header_filename, 'a') as headers_out
                ):
            while len(self.deq_data) > 0:
                ## Pop
                # Pop the oldest data
                data_chunk = self.deq_data.popleft()
                
                # Pop the oldest header
                data_header = self.deq_header.popleft()
            
            
                ## Set up the header row of headers_out if first time
                if self.header_colnames is None:
                    self.header_colnames = sorted(data_header.keys())
                
                    # Write the header
                    str_to_write = ','.join(self.header_colnames)
                    headers_out.write(str_to_write + '\n')            
                
                
                ## Append data to output file
                # Note: just maintains dtype of whatever data_chunk is
                data_out.write(data_chunk)
                
                # Append header to headers_out
                str_to_write = ','.join(
                    [str(data_header[colname]) 
                    for colname in self.header_colnames])
                headers_out.write(str_to_write + '\n')

                
                ## Keep track of how many chunks written
                self.n_chunks_written += 1
        
        if self.verbose:
            print(f"popped and wrote {self.n_chunks_written} chunks")

    def _target(self):
        """Target of the thread
        
        As long as self.keep_writing is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        Keep track of any late reads.
        
        Once self.keep_writing is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_writing
        while self.keep_writing:
            self.write_to_disk()
            
            # This sleeps keeps us from writing too frequently, which is 
            # probably somewhat expensive
            # But it increases the latency before draining
            time.sleep(.3)
        
        # self.keep_writing has been set False
        # Read any last chunks
        self.write_to_disk()
    
    def start(self):
        """Start the capture of data"""
        self.thread = threading.Thread(target=self._target)
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
        self.deq_header = collections.deque()
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
        self.deq_header.append(header)
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

