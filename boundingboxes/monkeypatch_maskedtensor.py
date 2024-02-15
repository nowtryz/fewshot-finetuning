import warnings
from functools import partial

import torch
from torch.masked.maskedtensor._ops_refs import (register_dispatch_func, _check_args_kwargs_length,  # noqa
                                                 _MASKEDTENSOR_DISPATCH_TABLE,  # noqa
                                                 register_function_func)  # noqa
from torch.masked.maskedtensor.core import _get_data, _maybe_get_mask, MaskedTensor, is_masked_tensor  # noqa
from torch.masked.maskedtensor.passthrough import PASSTHROUGH_FNS, _apply_pass_through_fn  # noqa


def monkeypatch_missing_dispatch_func(aten_ops):
    missing_ops = set(aten_ops) - set(_MASKEDTENSOR_DISPATCH_TABLE.keys())

    def wrapper(func):
        for aten_op in missing_ops:
            _MASKEDTENSOR_DISPATCH_TABLE[aten_op] = partial(func, aten_op)
    return wrapper


# See https://github.com/pytorch/pytorch/pull/125262
@monkeypatch_missing_dispatch_func([
    torch.ops.aten.unfold,
    torch.ops.aten.unfold_backward,
    # torch.ops.aten.stack,
])
def _general_passthrough(func, *args, **kwargs):
    return _apply_pass_through_fn(func, *args, **kwargs)


@monkeypatch_missing_dispatch_func([
    torch.ops.aten.stack,
])
def _general_passthrough(func, *args, **kwargs):
    return _apply_pass_through_fn(func, *args, **kwargs)


# See https://github.com/pytorch/pytorch/issues/128557
# See https://github.com/pytorch/pytorch/pull/128574
@monkeypatch_missing_dispatch_func([torch.ops.aten._is_any_true])  # noqa
def _is_any_true(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0)
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if mask is None:
        raise ValueError(f"__torch_dispatch__, {func}: expected a mask tensor")
    if data.dtype != torch.bool:
        raise ValueError(f"__torch_dispatch__, {func}: expected a boolean tensor")
    if data.is_sparse:
        raise ValueError(
           f"MaskedTensors with sparse data do not have {func}"
        )

    return MaskedTensor(func(data & mask), torch.tensor(True))


# See https://github.com/pytorch/pytorch/pull/128637
@monkeypatch_missing_dispatch_func([
    torch.ops.aten.zeros_like,
    torch.ops.aten.ones_like,
    torch.ops.aten.empty_like,
    torch.ops.aten.full_like,
    torch.ops.aten.rand_like,
    torch.ops.aten.randn_like,
    torch.ops.aten.randint_like
])
def _like_apis(func, *args, **kwargs):
    result_data = func(_get_data(args[0]), *args[1:], **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))


@monkeypatch_missing_dispatch_func([torch.ops.aten.masked_fill_])
def masked_fill_(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, "__torch_function__, torch.where", len_args=3, len_kwargs=0)
    self, mask, value = args
    func(_get_data(self), _get_data(mask), value)
