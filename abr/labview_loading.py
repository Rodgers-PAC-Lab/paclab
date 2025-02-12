"""Functions for loading ABR data saved by Bill's LabView code"""

import struct
import numpy as np


def parse_header(header):
    """Parse data from a single header

    `header`: 60 bytes

    Returns: dict, with the following items:
        packet_number : number of packet within the file
        number_channels : typically 8
        number_samples : this is the number of samples per packet
        trigger_position : whether a trigger occurred in this packet
            -1 if no trigger
            Otherwise, an index into the packet
        header_ver : typically 2
        total_pkts : total number of packets in the file
            This is only valid if it's the first header and if Labview
            completed the recording. Otherwise it's zero.
        sampling_sps : sampling rate
        gains : an array of channel gains
    """
    res = {}

    res['packet_number'] = struct.unpack('<I', header[0:4])[0]
    res['number_channels'] = struct.unpack('<I', header[4:8])[0]
    res['number_samples'] = struct.unpack('<I', header[8:12])[0]
    res['trigger_position'] = struct.unpack('<i', header[12:16])[0]
    res['header_ver'] = struct.unpack('<i', header[16:20])[0]
    res['total_pkts'] = struct.unpack('<I', header[20:24])[0]
    res['sampling_sps'] = struct.unpack('<i', header[24:28])[0]
    res['gains'] = np.array(struct.unpack('<8i', header[28:60]))

    return res

def parse_data(data_bytes, number_channels, number_samples):
    """Parse bytes into acquired data

    Takes the `data_bytes` that have been read from disk, removes all
    the header information, converts to float, concatenates across packets,
    and returns.

    data_bytes : bytes
        This is the data read from disk

    Returns: array of shape (n_samples, n_channels)
        This is all the data from each channel.
        The results are in volts, because that's how LabView saved them
    """
    # Each packet comprises a 60 byte headers (15 ints) and a 16000 byte data
    # payload (4000 floats)

    # It's easiest to convert everything to float, and then just dump the
    # floats that correspond to the headers

    # Unpack it all to floats
    n_floats = len(data_bytes) // 4

    # The old, slow way
    # data = np.array(struct.unpack('<{}f'.format(n_floats), data_bytes))

    # This is faster
    # https://stackoverflow.com/questions/36797088/speed-up-pythons-struct-unpack
    data = np.ndarray(
        (n_floats,), dtype=np.float32, buffer=data_bytes).astype(float)

    # Truncate to a multiple of 4015
    total_packet_size = 15 + number_channels * number_samples
    n_complete_packets = len(data) // total_packet_size
    if len(data) / total_packet_size != n_complete_packets:
        print("warning: data length was {}, not a multiple of packet length".format(len(data)))
        data = data[:total_packet_size * n_complete_packets]

    # Reshape into packets -- one packet (4015 bytes) per row
    data = data.reshape(-1, total_packet_size)

    # Drop the first 15 bytes of each row, which are the headers
    # TODO: instead of dropping, convert to ints, and extract header
    # of each packet
    data = data[:, 15:]

    # Each row needs to be reshaped into (number_channels, number_samples)
    # Note that the data is structured as sample-first;channel-second
    # which doesn't really make sense
    data = data.reshape((-1, number_channels, number_samples))

    # Concatenate all packets together (axis=1 accounts for the sample-first
    # problem). Transpose so each channel is a column
    data = np.concatenate(data, axis=1).T

    return data

def load_from_file(datafile, header_size=60):
    ## Open the file and read the header
    # Read the data
    with open(datafile, 'rb') as fi:
        data_bytes = fi.read()

    # Error if nothing
    if len(data_bytes) == 0:
        raise IOError("error: {} is empty".format(datafile))

    # We need to parse the first header separately because it has data that
    # we need to parse the rest of the packets.
    first_header_bytes = data_bytes[:header_size]
    first_header_info = parse_header(first_header_bytes)


    ## Parse the entire file
    # This appears to be in V
    data = parse_data(
        data_bytes,
        first_header_info['number_channels'],
        first_header_info['number_samples'])
    
    return data