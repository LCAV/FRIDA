from __future__ import print_function
import sys
import pkg_resources
import traceback as tb
from pkg_resources import DistributionNotFound, VersionConflict

# here, if a dependency is not met, a DistributionNotFound or VersionConflict
# exception is thrown. 
some_missing = False
with open('./requirements.txt', 'r') as f:
    dependencies = f.read().splitlines()

for dep in dependencies:
    try:
        pkg_resources.require([dep])
    except:
        print('Error: package', dep, 'is required.')
        some_missing = True

if some_missing:
    sys.exit(1)
else:
    print('All dependencies are satisfied.')
