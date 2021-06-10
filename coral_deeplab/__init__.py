__version__ = '0.2'

from . import pretrained
from ._downloads import from_precompiled

try:
    from . import (
        layers,
        applications
    )

except ImportError:
    print('coral-deeplab is running without tensorflow dependencies. '
          'You can still use it to pull precompiled models.')
