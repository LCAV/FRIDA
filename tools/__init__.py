__version__ = '1.0'

# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from .generators import *
from .plotters import *
from .utils import *
from .mkl_fft import *

