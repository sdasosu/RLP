"""Torchvision-backed model factories grouped by architecture family."""

from .vgg import __all__ as _vgg_all
from .vgg import *
from .resnet import __all__ as _resnet_all
from .resnet import *
from .mobilenet import __all__ as _mobilenet_all
from .mobilenet import *
from .densenet import __all__ as _densenet_all
from .densenet import *
from .googlenet import __all__ as _googlenet_all
from .googlenet import *
from .resnext import __all__ as _resnext_all
from .resnext import *
from .custom import __all__ as _custom_all
from .custom import *

__all__ = (
    list(_vgg_all)
    + list(_resnet_all)
    + list(_mobilenet_all)
    + list(_densenet_all)
    + list(_googlenet_all)
    + list(_resnext_all)
    + list(_custom_all)
)

del (
    _vgg_all,
    _resnet_all,
    _mobilenet_all,
    _densenet_all,
    _googlenet_all,
    _resnext_all,
    _custom_all,
)
