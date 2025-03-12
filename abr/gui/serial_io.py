class MultiprocSerialReader(object):
    def __init__(self, serial_port, serial_timeout, gains, sampling_rate, 
        session_dir, session_dt_str, verbose=True):
        # Store params
        self.serial_port = serial_port
        self.serial_timeout = serial_timeout
        self.gains = gains
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.session_dir = session_dir
        self.session_dt_str = session_dt_str
        
        # Output goes here
        self.output_data = multiprocessing.Queue()
        self.output_headers = multiprocessing.Queue()
        
        # Diagnostics
        self.late_reads = 0
        self.n_packets_read = 0
        
        # This is the flag used for stopping
        self.stop_event = multiprocessing.Event()
    
    def start_session(self):
        """Target of the thread
        
        This runs until stop_event is detected
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
        print('flushing serial port')
        
        # Flush
        self.ser.flushInput()
        self.ser.flush()
        
        # Stop data acquisition if it's already happening
        print('writing stop')
        serial_comm.write_stop(
            self.ser, warn_if_not_running=False, warn_if_running=True)
        
        
        ## Query the serial port
        # Just a consistency check because the answer should always be the same
        print('querying')
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
        
        
        ## Capture until we are told to stop
        while not self.stop_event.is_set():
            try:
                self.read_and_append()
            except Exception as e:
                print(f'ignoring {e}')
                pass
        
        print('stop event detected')
        
        ## Stop event was set, so stop
        self.stop_session()
        
        print('done with MultiprocSerialReader.start_session')

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
    
        # put_nowait means put right away, else raise an exception
        self.output_headers.put_nowait(data_packet['message'])
        self.output_data.put_nowait(data_packet['payload'])
        
        # Log
        self.n_packets_read += 1

    def stop_session(self):
        print('MultiProc stopping abr session...')

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
    

class ThreadedQueueReader(object):
    """Instantiates a thread to constantly read from multiproc qs into deques
    
    One deque will be provided to the GUI
    Another deque will be provided to the ThreadedFileWriter

    """
    def __init__(self, q_data, q_headers, verbose=False):
        """Instatiate a new ThreadedSerialReader
        
        ser : serial.Serial
            Data will be read from this object and stored in self.deq
        """
        self.q_data = q_data
        self.q_headers = q_headers
        self.keep_reading = True
        self.n_packets_read = 0
        self.thread = None
        self.verbose = verbose
        
        # This is a list of output deques
        self.deq_data_l = []
        self.deq_headers_l = []

        # The first one is always initialized (this is the one the GUI uses
        # Later ones are initialized on request
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
        self.deq_headers_l.append(new_deq_header)
        self.deq_data_l.append(new_deq_data)
        
        # Return
        return new_deq_header, new_deq_data
    
    def capture(self):
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
            
            ## Try to get a header
            try:
                header = self.q_headers.get_nowait()
            except multiprocessing.queues.Empty:
                header = None
            
            if header is not None:
                # Append to left side
                for deq_header in self.deq_headers_l:
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
            
            
            ## If we didn't get any data, sleep for a bit
            if header is None or data is None:
                time.sleep(0.3)
        
        # TODO: get any last chunks here
   
    def start(self):
        """Start the capture of data"""
        self.thread = threading.Thread(target=self.capture)
        self.thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_reading = False
        
        if self.thread is not None:
            self.thread.join()


