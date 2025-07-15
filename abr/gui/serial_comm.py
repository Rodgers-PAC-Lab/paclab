"""Functions to read and write to Teensy over the serial

"""
import time
import struct
import serial
import datetime
import numpy as np

## Functions to read Bill's protocol
def read_and_classify_packet(
    ser, wait_time=0.5, error_on_empty_message=True, assert_packet_type=None, 
    verbose=False):
    """Read a header and then read the appropriate packet
    
    ser: an open serial.Serial
    
    wait_time: float
        Maximum time to wait for message. If 0, it will only check once
        If no message is received before wait_time expires, the result is
        controlled by error_on_empty_message
    
    error_on_empty_message: bool
        if True and no data was received after ser timeout, then raise error
        if False and no data was received after ser timeout, then return None
    
    assert_packet_type: str or None
        if error_on_empty_message is False and no data was received, this has
        no effect
        if not None, the read packet is asserted to be of this type
    
    Returns: dict or None
        If no data was available, return None
        
        Otherwise:
        'packet_type_s': str
            One of 'query', 'ack', 'idle', or 'data'
        'message': dict
        'payload': array or None
    """
    
    ## Read until we get sync bytes or an error condition is reached
    # Determine how long we will wait to get sync bytes
    start_time = datetime.datetime.now()
    wait_time_threshold = start_time + datetime.timedelta(seconds=wait_time)
    
    # Read a partial_message -- ideally this is the sync bytes, but it could
    # be nothing, or something else
    sync_bytes = ser.read(4)

    # Keep trying until wait_time expires
    while datetime.datetime.now() < wait_time_threshold:
        if len(sync_bytes) < 4:
            # Message is only partial .. sleep and try again
            sync_bytes += ser.read(1)
        
        elif sync_bytes[-4:] == b'\xAA\x55\x5A\xA5':
            # Sync bytes found
            break
        
        else:
            # Data has been read but it wasn't sync bytes, so keep reading
            # A potential issue here if data is being sent faster than this
            # code can keep up with it, which will lead to late reads
            sync_bytes += ser.read(1)
    
    
    ## Deal with various edge cases relating to sync byte
    # Depends on how many sync bytes
    if len(sync_bytes) == 0:
        # No sync bytes
        # This might be normal if an empty_message is expected, otherwise
        # it's an error
        error_msg = (
            f'{start_time} sync error: incomplete sync message received: '
            f'{sync_bytes.hex()}')
        
        if error_on_empty_message:
            raise ValueError(error_msg)
        else:
            # This is the only case where a warning is not warranted
            return None
    
    elif len(sync_bytes) < 4:
        # 1-3 sync bytes - this should never happen
        # In this case, error or return None, since the following bytes won't
        # make sense
        error_msg = (
            f'{start_time} sync error: incomplete sync message received: '
            f'{sync_bytes.hex()}')

        if error_on_empty_message:
            raise ValueError(error_msg)
        else:
            print(error_msg)
            return None

    elif sync_bytes[-4:] != b'\xAA\x55\x5A\xA5':
        # Handle too many bytes received, and none of them sync bytes
        # In this case, error or return None, since the following bytes won't
        # make sense
        error_msg = (            
            f'{start_time} sync error: no sync bytes found out of '
            f'{len(sync_bytes)} total: {sync_bytes.hex()}')
        
        if error_on_empty_message:
            raise ValueError(error_msg)
        else:
            print(error_msg)
            return None
    
    elif len(sync_bytes) > 4:
        # Handle sync_bytes found, but only after extraneous bytes
        # In this case, warn about the dropped data and continue
        # This case, and the "good" case where exactly 4 correct sync bytes were
        # received, are the only cases where the code continues
        n_dropped_bytes = len(sync_bytes) - 4
        print(
            f'{start_time} warning: extra data found before sync bytes, '
            f'ignoring {n_dropped_bytes} bytes: {sync_bytes.hex()}')
    
    
    ## Define the return value
    # Query: 25 bytes, packet type 0e
    # Ack: 18 bytes, packet type 0b
    # Idle: 18 bytes, packet type 0c
    # Data: 18 bytes, packet type 07, followed by specified num of bytes (16000)
    packet_type_s = ''
    message = None
    payload = None
    
    
    ## Read the packet type
    packet_type = ser.read(1)

    if packet_type == b'\x0e':
        ## Query
        packet_type_s = 'query'
        
        # Error check
        if assert_packet_type is not None and assert_packet_type != packet_type_s:
            raise ValueError(
                'unexpected packet type: query instead of ' +
                str(assert_packet_type)
                )
        
        # Read 20 more bytes
        query_bytes = ser.read(20)
        
        # Parse
        message = parse_query_bytes(query_bytes)
    
    elif packet_type == b'\x0b':
        ## Ack
        packet_type_s = 'ack'
        
        # Error check
        if assert_packet_type is not None and assert_packet_type != packet_type_s:
            raise ValueError(
                'unexpected packet type: ack instead of ' +
                str(assert_packet_type)
                )

        # Read 13 more bytes
        ack_bytes = ser.read(13)
        
        # Parse
        message = parse_ack_bytes(ack_bytes)
    
    elif packet_type == b'\x0c':
        ## Idle
        packet_type_s = 'idle'
        
        # Error check
        if assert_packet_type is not None and assert_packet_type != packet_type_s:
            raise ValueError(
                'unexpected packet type: idle instead of ' +
                str(assert_packet_type)
                )
        
        # Read 13 more bytes
        idle_bytes = ser.read(13)
        
        # Parse
        message = parse_idle_bytes(idle_bytes)
    
    elif packet_type == b'\x07':
        ## Data
        packet_type_s = 'data'

        # Error check
        if assert_packet_type is not None and assert_packet_type != packet_type_s:
            raise ValueError(
                'unexpected packet type: data instead of ' +
                str(assert_packet_type)
                )

        # Read 13 more bytes
        data_header_bytes = ser.read(13)
        
        # Parse header
        message = parse_data_header_bytes(data_header_bytes)
        assert message['n_samples'] == 500
        assert message['n_channels'] == 8

        # Determine how long we will wait to get payload bytes
        start_time = datetime.datetime.now()
        wait_time_threshold = start_time + datetime.timedelta(seconds=0.2)
        
        # Read data payload
        payload_bytes = ser.read(16000)

        # TODO: Handle the case here where we never get a full packet
        while datetime.datetime.now() < wait_time_threshold:
            if len(payload_bytes) < 16000:
                # In most cases, we get 16000 or 0 bytes
                # But in some cases, we get only 4096
                payload_bytes += ser.read(16000 - len(payload_bytes))
                # print('re-reading')
            else:
                break
            
        # Parse data_payload_bytes
        if len(payload_bytes) == 16000:
            payload = parse_data_payload_bytes(payload_bytes)
        else:
            # I think that sometimes the data payload is incomplete and it
            # ends with an ACK message instead
            print(f'error: incomplete payload of len {len(payload_bytes)}, dropping')
            print(f'header bytes: {data_header_bytes}')
            print(f'header: {message}')
            if len(payload_bytes) > 36:
                print(f'last 36 bytes of payload: {payload_bytes[-36:].hex()}')
            payload = None
    
    else:
        ## TODO: Handle this, probably by scanning for more sync bytes
        # Could happen if sync bytes were previously scanned and
        # incorrectly found by random chance
        raise ValueError(f'unrecognized packet type: {packet_type}')
    
    
    ## Log latencies in the message
    done_time = datetime.datetime.now()
    message['start_time'] = str(start_time)
    message['done_time'] = str(done_time)
    message['time_taken'] = (done_time - start_time).total_seconds()
    
    # Return
    return {
        'packet_type_s': packet_type_s,
        'message': message,
        'payload': payload,
        }

def read_out_leftover_data(ser):
    """Read all leftover data from ser"""
    n_leftover_bytes = 0
    overflow = ser.read(10000)
    
    while len(overflow) > 0:
        print("Reading out leftover data in serial buffer ...")
        n_leftover_bytes += len(overflow)
        overflow = ser.read(10000)

    if n_leftover_bytes > 0:
        print(f'Read {n_leftover_bytes} leftover bytes\n')    
  
def slice_and_assert_equal(data, start, stop, assert_val=None, signed=False):
    """Slice data from start to stop and assert equal to assert_val
    
    data : bytestring
    start, stop : integer indices
    assert_val : int or None
        if not None, data[start:stop] must equal this
    signed : bool
    
    Returns: int
        int.from_bytes(data[start:stop], 'big', signed=signed)
    """
    # Note: when Python gets a byte string,
    # if you use an index bytestring[0] it'll interpret it as an integer,
    # but a slice bytestring[0:1] will be bytes
    assert start >= 0
    assert stop <= len(data)

    # Slice and convert to signed integer
    val = int.from_bytes(data[start:stop], 'big', signed=signed)
    
    # Assert
    if assert_val is not None and assert_val != val:
        raise ValueError(f'received {val} where {assert_val} was expected')
    
    # Return
    return val

def write_query(ser):
    """Queries the Teensy and returns metadata
    
    Sends the command b'Q' to the Teensy and decodes the results.
    """
    # Send a query request
    ser.write(b'Q')

    # Parse the results
    query_res = read_and_classify_packet(ser, assert_packet_type='query')

    # Return
    return query_res

def parse_query_bytes(query_bytes):
    """Parse the query bytes into a dict

    query_bytes : bytestring of length 25
        The bytes returned in response to a query message, excluding
        the sync bytes aa555aa5 and the packet_type 0e

    A typical value of query_bytes:
        b'\x13\x01\x00\x00\x01\x00\x01\x00\x08>\x80\x00\x18\x11\x94\xeel\x00\x00\x01' 
    
    which in hex is
        13 01 00 00 01 00 01 00 08 3e 80 00 18 11 94 ee 6c 00 00 01
    
    All of the values are interpreted as big-endian unsigned integers, 
    except for neg_fullscale which is signed.
    
    All of them are asserted to have the following fixed values, which should
    never change.
    
    byte indices    meaning         value               int(value)
    ------------    -------         -----               ----------
    byte 0          header_size     13                  19
    byte 1          data rep?       01                  1
    byte 2          null            00                  0
    bytes 3 - 5     device_id       00 01               1
    bytes 5 - 7     board_version   00 01               1
    bytes 7 - 9     nchan_max       00 08               8
    bytes 9 - 11    max_sps         3e 80               16000
    bytes 11 - 13   bit_depth       00 18               24
    bytes 13 - 15   pos_fullscale   11 94               4500
    bytes 15 - 17   neg_fullscale   ee 6c               -4500
    bytes 17 - 19   offset          00 00               0
    byte 19         checksum        01                  1

    Returns: dict
        Each entry in the meaning column above will be a key
        Each entry in the int(value) column above will be a value
    """    
    assert len(query_bytes) == 20
    
    # Initalize return dict
    res = {}
    
    # Store results
    res['header_size'] = slice_and_assert_equal(
        query_bytes, 0, 1, 19)
    res['data_representation'] = slice_and_assert_equal(
        query_bytes, 1, 2, 1)
    slice_and_assert_equal(
        query_bytes, 2, 3, 0) # null
    res['device_id'] = slice_and_assert_equal(
        query_bytes, 3, 5, 1)
    res['board_version'] = slice_and_assert_equal(
        query_bytes, 5, 7, 1)
    res['nchan_max'] = slice_and_assert_equal(
        query_bytes, 7, 9, 8)
    res['max_sps'] = slice_and_assert_equal(
        query_bytes, 9, 11, 16000)
    res['bit_depth'] = slice_and_assert_equal(
        query_bytes, 11, 13, 24)
    res['pos_fullscale'] = slice_and_assert_equal(
        query_bytes, 13, 15, 4500)
    res['neg_fullscale'] = slice_and_assert_equal(
        query_bytes, 15, 17, -4500, signed=True)
    res['offset'] = slice_and_assert_equal(
        query_bytes, 17, 19, 0)
    res['checksum'] = slice_and_assert_equal(
        query_bytes, 19, 20, 1)
    
    return res


def parse_ack_bytes(ack_bytes):
    """This is blank because I don't know what is contained in the ack message
    
    Returns: empty dict
        For compatibility with other parsing functions
    """
    # ACK is 18 bytes
    # 'aa555aa50b0c000800 e8 00000018000100 20'
    # After update I get this: spaces to designate changes
    # 'aa555aa50b0c000800 80 00000018000100 b8'
    
    # Presumably the other bytes don't mean anything?
    
    return {}
    
def parse_idle_bytes(idle_bytes):
    """This is blank because I don't know what is contained in the idle message
    
    Returns: empty dict
        For compatibility with other parsing functions
    """
    # 'aa555aa5 0c0c0008008000000018000100b9'
    
    return {}
    
def write_setupU(ser, cmd_str):
    """Write cmd_str to ser
    
    * Writes cmd_str
    * Reads ACK
    * Reads IDLE
    """
    # Write cmd_str
    ser.write(bytes(cmd_str, 'utf-8'))

    # Receive an ACK message
    ack_message = read_and_classify_packet(ser, assert_packet_type='ack')
    
    # Receive an IDLE message
    # For whatever reason, this takes about 0.5 s to happen
    idle_message = read_and_classify_packet(ser, assert_packet_type='idle')

def write_stop(ser, warn_if_running=False, warn_if_not_running=True, 
    verbose=False):
    """Write a stop message and deal with response"""
    # Log
    if verbose:
        print('beginning stop sequence')
    
    # Write cmd_str
    ser.write(b'X;')

    # Wait for ack
    # TODO: wait some maximum period of time here
    if verbose:
        print('waiting for ack')
    while True:
        received_packet = read_and_classify_packet(ser)
        
        if received_packet['packet_type_s'] == 'ack':
            break
        elif received_packet['packet_type_s'] == 'data' and received_packet['payload'] is None:
            # incomplete data packet, probably ending with an ack message and maybe an idle message
            print('breaking on incomplete data payload')
            break
        else:
            print(f"dropping packet of type {received_packet['packet_type_s']}")
    
    # Optionally receive an IDLE message
    if verbose:
        print('waiting for idle')
    idle_message = read_and_classify_packet(
        ser, assert_packet_type='idle', error_on_empty_message=False)

    # The IDLE is only sent if it was actually running before the X
    if warn_if_not_running and idle_message is None:
        print(
            'warning: no idle message returned, '
            'indicating device was unexpectedly not running')
    
    if warn_if_running and idle_message is not None:
        print(
            'warning: idle message returned, '
            'indicating device was unexpectedly running')

def write_start(ser, sleep_time=1):
    # Write cmd_str
    ser.write(b'S;')
    
    # Receive an ACK message
    ack_message = read_and_classify_packet(
        ser, assert_packet_type='ack')    

def parse_data_header_bytes(data_header_bytes):
    """Parse the data header bytes into a dict

    data_header_bytes : bytestring of length 13
        The header bytes of a data packet, excluding the sync bytes aa555aa5 
        and the packet_type 07

    A typical value of data_header_bytes:
        b'\x0c\x03\x01\x9a\x03\x01\xf4>\x80\xff\xff\x08m'
    
    which in hex is
        0c 03 01 9a 03 01 f4 3e 80 ff ff 08 6d

    All of the values are interpreted as big-endian unsigned integers.
    
    byte indices    meaning         value               int(value)
    ------------    -------         -----               ----------
    byte 0          header_size     0c                  12
    byte 1          data rep        03                  3
    byte 2          total_packets   01                  1
    byte 3          packet_num      __                  __
    byte 4          data_class      03                  3
    bytes 5 - 7     n_samples       01f4                500
    bytes 7 - 9     sampling_rate   3e 80               16000
    bytes 9 - 11    trig_pos        __ __               __
    bytes 11        n_channels      08                  8
    byte 12         checksum        __                  __
    
    
    Some explanation of the enums
    ---
    packet_type: 7 (07) for data, 11 (0b) for ACK, 12 (0c) for IDLE
    header_size: 12 (0c) for data (which is incorrect, it's 18), 19 for query 
    data_format: 3 for I32, 1 for query
    packet_num: increments with packet
        When this gets to 255, it rolls over to 0
    data_class: 3 means a 2D array
    n_samples: generally 500 (01f4) except for the last, incomplete packet
    trig_pos: ffff if no trig

    Returns: dict
        Each entry in the meaning column above will be a key
        Each entry in the int(value) column above will be a value
    """    
    assert len(data_header_bytes) == 13
    
    # Initalize return dict    
    res = {}
    
    # header_size is incorrect
    res['header_size'] = slice_and_assert_equal(
        data_header_bytes, 0, 1, 12)
    res['header_nbytes'] = 18

    # translate data_format
    res['data_format_enum'] = slice_and_assert_equal(
        data_header_bytes, 1, 2, 3)
    res['data_format'] = 'I32'

    res['total_packets'] = slice_and_assert_equal(
        data_header_bytes, 2, 3)
    
    res['packet_num'] = slice_and_assert_equal(
        data_header_bytes, 3, 4)
    
    # fixed at 3 for 2D array
    res['data_class'] = slice_and_assert_equal(
        data_header_bytes, 4, 5, 3)

    res['n_samples'] = slice_and_assert_equal(
        data_header_bytes, 5, 7)
    
    res['sampling_rate'] = slice_and_assert_equal(
        data_header_bytes, 7, 9, 16000)

    res['trig_pos'] = slice_and_assert_equal(
        data_header_bytes, 9, 11)
    
    res['n_channels'] = slice_and_assert_equal(
        data_header_bytes, 11, 12, 8)

    res['checksum'] = slice_and_assert_equal(
        data_header_bytes, 12, 13)
    
    return res
    
def quick_exit(ser):
    ser.write(b'X;')
    print(len(ser.read(100000)))
    ser.close()

def parse_data_payload_bytes(payload_bytes):
    # Unpack the payload
    # 'i' indicates int32, but the result is always int
    unpacked = struct.unpack('>4000i', payload_bytes)

    # reshape
    reshaped = np.reshape(unpacked, (500, 8))
    
    # convert back to int32, which is how it was loaded
    reshaped = reshaped.astype(np.int32)

    return reshaped
