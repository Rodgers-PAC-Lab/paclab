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
    
