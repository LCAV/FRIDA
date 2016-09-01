__version__ = '1.0'

# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from .compactsix_circular_1 import R_compactsix_circular_1
from .compactsix_random_1 import R_compactsix_random_1
from .pyramic_tetrahedron import R_pyramic

arrays = {
        'compactsix_circular_1': R_compactsix_circular_1,
        'compactsix_random_1': R_compactsix_random_1,
        'pyramic_tetrahedron': R_pyramic,
        }
