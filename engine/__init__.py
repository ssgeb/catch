"""Engine module initialization"""

# for register purpose
from . import optim
from . import data
from . import catch

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
