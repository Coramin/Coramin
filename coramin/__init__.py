import sys
if sys.version_info.major != 3 or sys.version_info.minor < 6:
    raise EnvironmentError('Coramin only supports Python 3.6 and newer.')

from coramin import utils
from coramin import domain_reduction
from coramin import relaxations
