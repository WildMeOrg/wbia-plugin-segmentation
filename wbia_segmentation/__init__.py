# -*- coding: utf-8 -*-
from wbia_segmentation import _plugin  # NOQA

try:
    from wbia_segmentation._version import __version__  # NOQA
except ImportError:
    __version__ = '0.0.0'
