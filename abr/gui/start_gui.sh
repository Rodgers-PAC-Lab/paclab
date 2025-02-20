#!/bin/bash
# This must be run with "bash -i" for conda to work

# -u makes python run in unbuffered mode so we can see error messages
conda activate py3 && python3 -u -m paclab.abr.gui.start_gui |& tee -a /home/mouse/mnt/cuttlefish/surgery/abr_logfiles/logfile.$(date '+%F-%T-%N')

echo ABR GUI closed. Press any key to close this window. 
echo PAC Lab personnel - do not ignore this message\! Close this window\!
read
