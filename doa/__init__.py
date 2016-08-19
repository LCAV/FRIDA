__version__ = '1.0'

# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from .doa import *
from .srp import *
from .music import *
from .cssm import *
from .waves import *
from .tops import *
from .fri import *

import tools_fri_doa_plane as tools_fri

# Create this dictionary as a shortcut to different algorithms
algos = {
        'SRP' : SRP,
        'MUSIC' : MUSIC,
        'CSSM' : CSSM,
        'WAVES' : WAVES,
        'TOPS' : TOPS,
        'FRI' : FRI,
        }

