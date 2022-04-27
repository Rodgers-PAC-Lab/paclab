"""Methods for setting the path based on the computer hostname"""
import socket
import os

def get_path_to_terminal_data():
    """Identify path to autopilot data based on computer hostname
    
    Currently this is set to the "from_clownfish" directory instead of the
    old "from_octopus" directory.
    
    Raises IOError if that path does not exist.
    
    Right now this could be in a different user's home directory on each
    computer. And the path is different on cephalopod. Might want to
    normalize this across all computers.
    """
    # Check to see which computer it's running on
    computer = socket.gethostname()
    
    # Generate path
    if computer == 'cephalopod':
        path_to_terminal_data = (
            '/home/chris/mnt/cuttlefish/behavior/from_clownfish/autopilot'
            '/terminal/autopilot/data')
    elif computer == 'octopus':
        path_to_terminal_data = (
            '/home/mouse/mnt/cuttlefish/from_clownfish/autopilot'
            '/terminal/autopilot/data')
    else:
        path_to_terminal_data = (
            '/home/rowan/mnt/cuttlefish/from_clownfish/autopilot'
            '/terminal/autopilot/data')
    
    # Check that the path exists
    if not os.path.exists(path_to_terminal_data):
        raise IOError(
            "error, terminal data is supposed to be at "
            "{} but it doesn't exist".format(path_to_terminal_data))
    
    # Return
    return path_to_terminal_data
