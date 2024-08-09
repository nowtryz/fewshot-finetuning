import functools
import inspect
from typing import Callable, ParamSpec

import numpy as np
import torch
from monai.data import MetaTensor


def auto_repr(cls: type) -> type:
    """
    Auto generates the ``__repr__`` on the decorated class.

    :param cls: Class to be decorated.
    """
    if not hasattr(cls, '__repr__'):
        def __repr__(self):
            params = (f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_'))
            return f"{self.__class__.__name__}({', '.join(params)})"

        cls.__repr__ = __repr__
    return cls
