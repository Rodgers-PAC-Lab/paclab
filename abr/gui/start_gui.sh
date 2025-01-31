#!/bin/bash

# This must be run with "bash -i" for conda to work

conda activate py3 && python3 -m paclab.abr.gui.start_gui

echo ABR GUI closed. Press any key to close this window. 
echo PAC Lab personnel - do not ignore this message\! Close this window\!
read
