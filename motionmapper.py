# Helper functions for motionmapperpy
import tomllib
import easydict
import os

def load_mmpy_parameters(filename=None):
    """Load motionmapperpy parameters from a TOML file
    
    filename : full path to TOML file
    
    The TOML file is grouped into families, but for backwards compatibility,
    all the dicts are concatenated into a single one. 
    
    Returns: dict
    """
    # Default is ./default.toml
    if filename is None:
        this_dir = os.path.split(__file__)[0]
        filename = os.path.join(this_dir, 'default.toml')
    
    # Load from TOML
    with open(filename, 'rb') as fi:
        parameters = tomllib.load(fi)

    # Flatten into a single dict to match old style
    res = {}
    for family, family_dict in parameters.items():
        for key, val in family_dict.items():
            if key in res:
                print(f'warning: overwriting duplicate key {key} in {filename}')
            res[key] = val
    
    # Convert to easydict
    res = easydict.EasyDict(res)
    
    return res