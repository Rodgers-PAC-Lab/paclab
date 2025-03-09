import multiprocessing
import threading
import collections
import time

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
        self.mpq = multiprocessing.Queue()
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
        
        # When the buffer fills, what seems to happen is that the next packet
        # is fine, then the following packet starts out fine but ends corrupted,
        # then the following packet starts out corrupted, then we get back
        # on track. So probably the buffer can hold >1 but <2 packets, and
        # when the second packet is read, the end of it is some random chunk
        # of another packet.
        # The buffer might be 20132 (16018 + 4114). 
        # After the first packet is read, the second packet is only 4114 long, 
        # then it reads 11904 from the next packet to arrive, then the last
        # 4114 of that packet are dropped.

        #~ # Debug: intentionally break
        #~ if self.n_packets_read == 10:
            #~ time.sleep(.5)

    def dummy_read(self):
        # Append to the left side of the deque
        self.deq_headers.append(self.n_packets_read)
        self.deq_data.append(self.n_packets_read)
        self.mpq.put(self.n_packets_read)
        
        # Log
        self.n_packets_read += 1
        
        print(f'dummy read {self.n_packets_read}')
        if self.n_packets_read > 20:
            1/0
        time.sleep(.1)
    
    def capture(self):
        """Target of the thread
        
        As long as self.keep_reading is True, infinitely keep reading
        chunks of data and appending to the deq as they come in.
        Keep track of any late reads.
        
        Once self.keep_reading is False, read any last chunks, 
        and then return.
        """
        # Continue as long as self.keep_reading
        while not self.stop_event.is_set():
            if self.verbose:
                print(f'deqlen: {len(self.deq_data)}')
            #~ self.read_and_append()
            self.dummy_read()
            
            #~ if self.n_packets_read == 10:
                #~ print('simulating pause')
                #~ # It seems like after this pause we get the next two packets, then a
                #~ # gap, then packets continue. But never a partial packet.
                #~ # Perhaps two packets fills the buffer, and after that all messages are
                #~ # simply dropped silently until there is space again.
                #~ time.sleep(1)
        
        #~ # self.keep_reading has been set False
        #~ # Read any last chunks
        #~ while self.ser.in_waiting > 0:
            #~ # Probably should only do this once, because if it's more than
            #~ # one, something has gone wrong 
            #~ #self.read_and_append()
            #~ self.dummy_read()
        
        print('done')
    
    def start(self):
        """Start the capture of data"""
        #~ self.thread = threading.Thread(target=self.capture)
        #~ self.thread.start()
        
        self.stop_event = multiprocessing.Event()
        self.thread = multiprocessing.Process(target=self.capture)
        self.thread.start()
    
    def stop(self):
        """Stop capturing data and join the thread"""
        self.keep_reading = False
        self.stop_event.set()
        
        if self.thread is not None:
            self.thread.join(timeout=1)


tsr = ThreadedSerialReader(ser=None)


tsr.start()
time.sleep(1)
tsr.stop()


