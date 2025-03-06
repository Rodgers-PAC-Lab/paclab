# Need to have permission to access ACM0
# sudo adduser mouse dialout
#
# run this like:
#   python3 -m paclab.abr.gui.start_cli
# or in ipython --pylab=qt:
#   run -m paclab.abr.gui.start_cli

import numpy as np
import pandas
from . import ABR_Device
import paclab
import os

# Initalize abr_device
abr_device = ABR_Device.ABR_Device()

# Run
abr_device.run_session()

# Load data
loaded_data = paclab.abr.loading.load_recording(
    os.path.split(abr_device.tfw.output_filename)[0])

config = loaded_data['config']
header_df = loaded_data['header_df']
data = loaded_data['data']

if np.any(np.diff(header_df['packet_num_unwrapped']) != 1):
    print('warning: data is torn')
    
"""
Debugging stuff
If you add time.sleep to read_and_append, you can get it to drop a bunch 
of packets and tear data, but it will get back on track.
But that doesn't really simulate the GUI problem. It would be a good way
to test the effects of raising the USB buffer size, but that doesn't seem
possible anyway.

Option 1
- Add some sleeps to read_and_classify_packet to make it tear. Then double
  buffer the serial port reader. This should become resilient to sleeps in
  read_and_classify_packet. Unclear if it will solve the GUI problem, since
  the GUI might block even the double-buffered reader. We don't know if
  the problem only affects slow processing of payloads, or something else.

Option 2
- Move serial read to a multiprocess. Then test GUI. It might work right away,
  if the underlying problem is threading. The downside is there is no simple 
  way to test this other than running the full-blown GUI.

Plan:
- Let's try the multiprocess first as it seems fundamentally the better option.
"""