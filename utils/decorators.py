import functools
import inspect
from typing import Callable, ParamSpec

import numpy as np
import torch
from monai.data import MetaTensor


P = ParamSpec('P')


def wrap_ndarray_in_tensor(function: Callable[P, np.ndarray]) -> Callable[P, torch.Tensor]:
    """
    Decorator that converts a Tensor given as function's input to a numpy nd-array and converts the result from the
    function back to a tensor. Basically wraps a numpy function to its pytorch equivalent.

    :param function: Function to wrap
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> torch.Tensor:
        args = [arg.numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {key: arg.numpy() if isinstance(arg, torch.Tensor) else arg for key, arg in kwargs.items()}
        result = function(*args, **kwargs)
        return torch.as_tensor(result)

    return wrapper


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


def preserves_meta(key: str):
    """
    Automatically copy metadata from a `MetaTensor` received as input and cast the `torch.Tensor` return by the function
    to a `MetaTensor` filled with the gather metadata.

    :param key: the name of the argument to copy the metadata from.
    """
    def _decorator(function: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        sig = inspect.signature(function)

        @functools.wraps(function)
        def _wrapper(*args, **kwargs) -> torch.Tensor:
            output = function(*args, **kwargs)
            binding = sig.bind(*args, **kwargs).arguments
            input_ = binding[key]
            if isinstance(input_, MetaTensor) and not isinstance(output, MetaTensor):
                return MetaTensor(output, meta=input_.meta, applied_operations=input_.applied_operations)
            return output

        return _wrapper
    return _decorator
