import warnings
from functools import partial
from itertools import accumulate
from typing import Sequence, Union, List, Tuple

import torch
from torch import Tensor
from torch.utils.data._utils.collate import (  # noqa: as per docs, collate must be import through the private module
    collate, default_collate_fn_map, collate_tensor_fn)


def sparse_pad_sequence_(sequences: Sequence[torch.Tensor], batch_first=False, dim=0):
    """
    Stacks a sequence of tensor to build a single tensor, padding all input tensors to match the size of the biggest
    one.

    This operation involves inplace operations by resizing the tensors given as input before staching them. As a result,
    this implementation is faster than `sparse_pad_sequence` but the inplace operation must be taken into account.
    """
    sequences_size = len(sequences)
    assert sequences_size > 0, "received an empty list of sequences"
    assert all(sequence.is_sparse for sequence in sequences), "expected sparse COO tensors"

    if __debug__ and any(t.requires_grad for t in sequences):
        warnings.warn('Received tensor requiring a gradient to `sparse_pad_sequence_` but in place operation will'
                      'occur. Consider using `sparse_pad_sequence` instead.')

    max_len = max(sequence.size(dim) for sequence in sequences)
    size_0 = sequences[0].size()
    new_size = torch.Size((*size_0[:dim], max_len, *size_0[dim+1:]))

    for sequence in sequences:
        sequence.sparse_resize_(new_size, sequence.sparse_dim(), sequence.dense_dim())

    # torch.stack expect either a tuple or a list of tensors
    if not isinstance(sequences, (tuple, list)):
        sequences = list(sequences)

    return torch.stack(sequences, 1 if batch_first else 0)


def sparse_pad_sequence(sequences: Sequence[torch.Tensor], batch_first=False):
    """
    TODO documentation

    .. note::
        Slower than the original implementation but leverage the power of the existing scarcity present in the inputs to
        build a padded sparse result. Resulting in less memory used for computation and possibly faster computation that
        the native pad_sequence implementation in some edge cases.
    """
    # Inspired by torch's aten/src/ATen/native/PackedSequence.cpp but supports sparse tensors
    sequences_size = len(sequences)

    assert sequences_size > 0, "received an empty list of sequences"
    assert sequences[0].layout == torch.sparse_coo, "expected sparse COO tensors"

    trailing_dims = sequences[0].size()[1:]
    num_sparse_dims = sequences[0].sparse_dim() + 1
    max_len = max(sequence.size(0) for sequence in sequences)

    out_dims = (max_len, sequences_size) if batch_first else (sequences_size, max_len)
    out_dims += trailing_dims

    num_specified_elements = [sequence.indices().size(1) for sequence in sequences]
    indices = torch.zeros(num_sparse_dims, sum(num_specified_elements))

    iterator = zip(
            enumerate(sequences),
            num_specified_elements,
            accumulate(num_specified_elements, initial=0)
    )
    if batch_first:
        for (seq_index, sequence), nse, start_index in iterator:
            sequence_indices = sequence.indices()
            indices[0, start_index:start_index + nse] = sequence_indices[0]
            indices[1, start_index:start_index + nse] = seq_index
            if num_sparse_dims > 2:
                indices[2:, start_index:start_index + nse] = sequence_indices[1:]
    else:
        for (seq_index, sequence), nse, start_index in iterator:
            indices[0, start_index:start_index+nse] = seq_index
            indices[1:, start_index:start_index+nse] = sequence.indices()

    return torch.sparse_coo_tensor(
        indices=indices,
        values=torch.cat([sequence.values() for sequence in sequences]),
        size=out_dims,
    )


def collate_sparse_tensor_fn(batch: Sequence[torch.Tensor], *, collate_fn_map=None, pad_dim=0):
    elem = batch[0]
    if elem.is_sparse:
        # TODO
        #  - allow to choose between pad_sequence and sparse_pad_sequence in the transformation
        #  - add a try catch around pad_sequence and automatically switch to sparse_pad_sequence and warn in case of
        #    memory issues
        return sparse_pad_sequence_(batch, dim=pad_dim).coalesce()

    return collate_tensor_fn(batch)


def sparse_collate(batch):
    """TODO"""
    return collate(batch, collate_fn_map={
        **default_collate_fn_map,
        # We want to pad the second dimension, being the boxes dimension. This only covers the case of our bonding boxes
        # but is good enough for the project.
        torch.Tensor: partial(collate_sparse_tensor_fn, pad_dim=1),
    })


def sparse_list_data_collate(batch):
    """TODO"""
    data = [e for samples in batch for e in samples] if isinstance(batch[0], list) else batch
    return sparse_collate(data)


def _torch_sparse_sum_dense_dim(tensor: Tensor, dim: Union[int, List[int], Tuple[int, ...]]) -> Tensor:
    """
    Allows to sum on dense dimensions, keeping the original dtype. Workaround for
    https://github.com/pytorch/pytorch/issues/122711
    """
    tensor = tensor.coalesce()
    assert isinstance(dim, int) and dim < 0 or all(d < 0 for d in dim), "Must provide negative dims"
    new_values = tensor.values().sum(dim)
    return torch.sparse_coo_tensor(
        tensor.indices(), new_values,
        tensor.shape[:tensor.sparse_dim()] + new_values.shape[1:]
    )


def _torch_sparse_sum_all_dense_dim(tensor: Tensor) -> Tensor:
    """
    Allows to sum on dense dimensions, keeping the original dtype. Workaround for
    https://github.com/pytorch/pytorch/issues/122711
    """
    tensor = tensor.coalesce()
    new_values = tensor.values().sum(tuple(range(1, tensor.dense_dim() + 1)))
    return torch.sparse_coo_tensor(tensor.indices(), new_values, tensor.shape[:tensor.sparse_dim()])
