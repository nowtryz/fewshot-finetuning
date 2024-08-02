r"""
The present module constitute all the losses used to handle supervising with bounding as well as the
`LogBarrierPenalty`. It is primarily based on the work of Hoel Kervadec from `Bounding boxes for weakly supervised
segmentation: Global constraints get close to full supervision`__. We deliberately restructure the code to match the
training engines used in this project and for it to be more adaptable and performant.

Note:
    Please note that this module heavily relies on sparce and masked tensors from `torch.sparce` and `torch.mask`
    respectively. Those allow to store all the bonding boxes in a memory efficient format while keeping computation as
    clear as possible. Reader are encouraged to ignore code involving masking and sparse computation and treat the code
    as if it was handling dense (normal) tensors. Advanced users may still check the respective documentation to have a
    better understanding of the mechanics. Here are a list of functions to ignore:
     * `torch.sparse.sum`, equivalent to `torch.sum`
     * `torch.Tensor.sparse_mask`, equivalent to `torch.Tensor.masked_scatter`
     * `as_masked_tensor`
     * `MaskedTensor.to_tensor`
"""
import functools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Union, Callable, cast, Literal, Tuple, TypeAlias

import torch
from torch import nn
from torch.masked import MaskedTensor

from . import monkeypatch_maskedtensor  # noqa: fix missing ops
from .sparse_utils import _torch_sparse_sum_dense_dim, _torch_sparse_sum_all_dense_dim

# Images dimensions from the bonding boxes masks and logits (when in 3D)
H, W, D = -3, -2, -1
H_2D, W_2D = -2, -1


def as_masked_tensor(data, mask):
    """
    Overload original `torch.masked.as_masked_tensor` to convert masked tensors back to normal tensors in the backward
    pass. Filling missing gradients with 0.
    """

    class Constructor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, data, mask):  # noqa
            return MaskedTensor(data, mask)

        @staticmethod
        def backward(ctx, grad_output):  # noqa
            return grad_output._masked_data.masked_fill(~grad_output.get_mask(), 0), None  # noqa

    result = Constructor.apply(data, mask)
    return result


def _mask_logits(logits: torch.Tensor, box_masks: torch.Tensor) -> torch.Tensor:
    """
    Broadcast our logits to the same dimension of the logits and mask them with our box masks to get a sparse tensor
    with the same dimension as the mask to allow element-wise multiplication, then mask them with the provided box
    masks.

    :param logits: Dense tensor in the ``B x C x Im...`` dimensions, representing the predicted segmentation
    :param box_masks: Sparse tensor in the ``B x C x N x im...`` dimensions representing the box labels
    :returns: The masked logits with the box mask
    """
    # Broadcast logits to the same dimensions as the box_masks to allow element-wise multiplication
    masked_logits = (
        logits
        .unsqueeze(dim=2)
        .broadcast_to(box_masks.size())
        .sparse_mask(box_masks)
    )  # B x C x N x Im...

    # mask logits for all boxes
    # Shape B x C x N x Im...
    return masked_logits * box_masks


class Reduction(Enum):
    MEAN = 'mean'
    SUM = 'sum'
    ORIGINAL = 'original'
    NONE = 'none'


class Mode(Enum):
    MODE_3D = '3D'
    MODE_2D = '2D'


ReductionArg: TypeAlias = Union[Reduction, Literal['mean', 'sum', 'original', 'none']]


class LogBarrierExtension(torch.autograd.Function):
    r"""
    Computes the *log-barrier extension* :math:`\tilde \psi_t (z)` and its derivative.

    Given the definition:

    .. math::

        \tilde\psi_t(z) =
        \begin{cases}
            -\frac{1}{t} \log(-z)                             & \text{if } z\leq -\frac{1}{t^2}, \\
            tz-\frac{1}{t} \log(\frac{1}{t^2}) + \frac{1}{t}  & \text{otherwise}
        \end{cases}

    and the derivative:

    .. math::

        \frac{\partial \tilde \psi_t (z)}{\partial x} =
        \begin{cases}
          -\frac{1}{tz} & \text{if } z\leq -\frac{1}{t^2}, \\
          t             & \text{otherwise}
        \end{cases}

    """

    @staticmethod
    def forward(ctx, z, t):
        cond = z <= - 1 / t ** 2
        ctx.mark_non_differentiable(t)
        ctx.save_for_backward(cond, z, t)
        return torch.where(cond,
                           - torch.log(-z) / t,
                           t * z - torch.log(1 / t ** 2) / t + 1 / t)

    @staticmethod
    def backward(ctx, grad_output):
        cond, z, t = ctx.saved_tensors  # type: torch.Tensor
        return torch.where(cond, - 1 / (t * z), t) * grad_output, None


class LogBarrierPenalty(nn.Module):
    r"""
    *Log-barrier extension* for Lagrangian optimization as defined by `H. Kervadec et al.`__.

    __ https://arxiv.org/abs/1904.04205

    .. math::

        \tilde\psi_t(z) =
        \begin{cases}
            -\frac{1}{t} \log(-z)                             & \text{if } z\leq -\frac{1}{t^2}, \\
            tz-\frac{1}{t} \log(\frac{1}{t^2}) + \frac{1}{t}  & \text{otherwise}
        \end{cases}

    :param t: Extension parameter.
        Defines how the extension is similar to the standard log-barrier. "When :math:`t \to +\infty`, [the] extension
        can be viewed [as] a smooth approximation of hard indicator function :math:`H`"
    :param epoch_multiplier: TODO
    """
    _t: torch.Tensor
    _epoch_multiplier: torch.Tensor
    _b: torch.Tensor
    _ceil: torch.Tensor

    def __init__(self, t: float = 5., epoch_multiplier: Union[int, float] = 1.1):
        super().__init__()
        self.register_buffer('_t', None)
        self.epoch_multiplier = epoch_multiplier
        self.t = t

    def extra_repr(self):
        return f't={self.t}, epoch_multiplier={self.epoch_multiplier}'

    @property
    def t(self) -> float:
        """Extension parameter, see parameter ``t`` from `LogBarrierPenalty`"""
        return self._t.item()

    @t.setter
    def t(self, value):
        self._t = torch.as_tensor(value, dtype=torch.float).to(self._t)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return LogBarrierExtension.apply(z, self._t)

    def step(self):
        self.t *= self.epoch_multiplier


class InequalityL2Penalty(nn.Module):
    @staticmethod
    def forward(value: torch.Tensor) -> torch.Tensor:
        return torch.where(cast(torch.Tensor, value >= 0), torch.square(value), 0)


class BoxTightnessPriorLoss(nn.Module):
    r"""
    Ensure the assumption that a slice of thickness :math:`w` taken from a bounding box, must have at least :math:`w`
    voxels predicted inside it. This loss is based and derived from the work from `H. Kervadec et al.`__ that only
    supports 2-Dimensional images. The 3-Dimensional adaptation uses "slices" instead of "segments"

    __ https://arxiv.org/abs/2004.06816

    .. math:: \sum_{v \in s_l} y_v \ge w & \forall s_l \in S_L

    For voxels :math:`v` in slices :math:`s_l` where :math:`S_L = \{ s_l \}` is a set of all slices of thickness
    :math:`w` parallel to the sides of the bounding box in any of the 3 dimensions.

    :param slices_width: Thickness of the slices (or segments in 2D)
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'```|
        ``'original'``. ``'original'`` means the original implementation of the paper: sum of all errors, for all boxes,
        images, and batch, while in the other cases the boxes are averaged and the reduction is applied over the last 2
        dimension (:math:`B` and :math:`C`)
    """

    def __init__(self, slices_width: int, penalty: Callable = None, reduction: ReductionArg = 'original',
                 mode: Mode = Mode.MODE_3D):
        super().__init__()
        self.slices_width = slices_width
        self.penalty = penalty if penalty is not None else InequalityL2Penalty()
        self.reduction = Reduction(reduction)
        self.mode = mode
        self._slice_dims = [(H, W), (W, D), (H, D)] if mode is Mode.MODE_3D else [(H_2D,), (W_2D,)]

    def extra_repr(self):
        return (f'slices_width={self.slices_width}, '
                f'reduction={self.reduction}, '
                f'mode={self.mode}')

    def forward(self, logits: torch.Tensor, box_masks: torch.Tensor):
        r"""
        Computes the Box Tightness Prior loss from the given logits and bonding box masks.

        :param logits: Softmax logits predicted by the model
        :param box_masks: Masks containing all the bounding boxes for each classes

        Shape:
             - ``logits``: :math:`(B \times C \times W \times H \times D)`
             - ``box_masks``: :math:`(B \times C \times N \times W \times H \times D)` with sparse dimensions :math:`B,
               C, N`

            With :math:`B`, :math:`C`, :math:`N` and :math:`(W, H, D)` being respectively the batch size, the number of
            classes, the maximum number of bonding boxes and the images dimensions.
        """
        assert box_masks.is_sparse, ("`box_masks` is expected to be supplied as sparse COO tensor, please call "
                                     "`box_masks = box_masks.to_sparse(3)`")

        predicted_boxes = _mask_logits(logits, box_masks)

        # Check if some boxes are present in the mask
        if box_masks.size(dim=2) == 0:
            # If no box is present, no need to compute the loss, we'll just return a loss of 0 as there is no
            # information to use for this specific loss.
            # We still need to yield a result computed from the given logits to connect the gradient graph
            # OPTIMIZATION NOTE: N-dim=0, dense-sum has 0 value to sum up
            if self.reduction is Reduction.NONE:
                return predicted_boxes.to_dense().sum(tuple(range(3, box_masks.dim())))
            return predicted_boxes.to_dense().sum()

        # Compute and evaluate slices in each of the 3 dimensions or the segments in each of the 2 dimensions
        errors = []
        segment_counts = []
        for slice_dims in self._slice_dims:
            # Shape B x C x N x result of slices (with lots of 0)
            slices = torch.sparse.sum(predicted_boxes, dim=slice_dims).to_dense()

            # We need to carry a mask with the same computation to then create our masked tensor as torch.Tensor.unfold
            # is not supported by masked tensors
            # Bug https://github.com/pytorch/pytorch/issues/122711
            # mask = torch.sparse.sum(box_masks, dim=slice_dims).to_dense().bool()
            # Workaround:
            mask = _torch_sparse_sum_dense_dim(box_masks, dim=slice_dims).to_dense().bool()

            # /!\ Computations are done on masked tensors to discriminate 0 from masking and 0 from the logits
            slices = as_masked_tensor(slices, mask)

            # TODO pad slices mean (on image border, image_dim % slices_width are removed)
            # TODO maybe replace unfold with view after padding
            # Create a sliding window of `slices_width` and average over the window to compute prediction per "segments"
            # Gives how many px predicted averaged per slice, on each "segment", taking of borders edge cases
            # Shape B x C x N x segments (with lots of 0)
            slices = slices.unfold(dimension=3, size=self.slices_width, step=self.slices_width)
            slices_mean = slices.mean(dim=-1)
            segment_counts.append(slices_mean.get_mask().sum(dim=-1))  # Shape B x C x N

            # On each "segment", the average px per slice must be >= 1
            slices_error = -slices_mean + 1
            # Fill masked values with 0s as the masked computation is done (segments)
            slices_error = slices_error.to_tensor(0)
            # We just want a loss when there is no voxel on a slice, skip negative values by replacing them with 0s
            slices_error = torch.maximum(slices_error, torch.zeros_like(slices_error))
            # Reduce errors per bounding boxes
            error = slices_error.sum(dim=-1)  # Shape B x C x N
            errors.append(error)

        # Concatenate error tensors in a new dimension (first dim)
        # Multiply by width to overcome `slices_mean = slices.mean(dim=-1)` and match original implementation, enforces
        # an implementation yielding the same result as the original implementation (with minimal differences)
        error = torch.stack(errors, dim=-1).sum(dim=-1) * self.slices_width  # Shape B x C x N
        segment_counts = torch.stack(segment_counts, dim=-1).sum(dim=-1)  # Shape B x C x N

        # Apply penalty
        error = self.penalty(error)
        assert error.size() == box_masks.size()[:3], 'Penalty should not change the dimensions of the tensor'

        # Reduce loss and return
        if self.reduction is Reduction.SUM:
            return torch.sum(error) / (torch.sum(segment_counts) or 1 / self.slices_width)
        if self.reduction is Reduction.ORIGINAL:
            return torch.sum(error)
        # Each `slices_error` has values between 0 and 1 and the number of segments is the dimension of the slice
        # divided by the width of the slice (neglecting edges, which are smaller)
        # Replace 0s with 1s in the `segment_counts` tensor to avoid divisions by 0
        divisor = segment_counts.where(cast(torch.Tensor, segment_counts != 0), 1)
        error = error / divisor / self.slices_width
        if self.reduction is Reduction.MEAN:
            return torch.sum(error) / (torch.count_nonzero(segment_counts) or 1)
        return error  # reduction == 'none'


class BoxSizePriorLoss(nn.Module):
    r"""
    Box Size loss from `H. Kervadec et al.`__, penalising the model when it predicts a box soft-size outside the
    proportion given as ``minimum`` and ``maximum``.

    __ https://arxiv.org/abs/2004.06816

    .. math:: \alpha \le \sum_{p \in \Omega_I} s_{\theta}(p) \le \beta

    With :math:`\alpha` and :math:`\beta` being respectively the minimum and maximum bounds and :math:`\Omega_{I_i}` the
    set of pixels inside the bonding box :math:`i`, for each bonding box.

    :param minimum: Minimum proportion of the box size the model should predict (value between 0 and 1).
    :param maximum: Maximum proportion of the box size the model should predict (value between 0 and 1).
    :param penalty: Penalty to apply before the reduction when the error has been computed
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'```|
        ``'original'``. ``'original'`` means the original implementation of the paper: sum of all errors divided by the
        number of pixels/voxels in the input image.
    """
    minimum: torch.Tensor
    maximum: torch.Tensor

    def __init__(self, minimum: Union[float, torch.Tensor], maximum: Union[float, torch.Tensor],
                 penalty: Callable[[torch.Tensor], torch.Tensor] = None, reduction: ReductionArg = 'original'):
        super().__init__()
        self.register_buffer('minimum', torch.as_tensor(minimum, dtype=torch.float))
        self.register_buffer('maximum', torch.as_tensor(maximum, dtype=torch.float))
        self.penalty = penalty if penalty is not None else InequalityL2Penalty()
        self.reduction = Reduction(reduction)

        assert self.minimum.size() == (), '`minimum` and `maximum` are supposed to be scalars'
        assert self.maximum.size() == (), '`minimum` and `maximum` are supposed to be scalars'

    def extra_repr(self):
        return (f'minimum={self.minimum}, '
                f'maximum={self.maximum}, '
                f'reduction={self.reduction}')

    def forward(self, logits: torch.Tensor, box_masks: torch.Tensor):
        r"""
        Computes the Box Size loss from the given logits and bonding box masks.

        :param logits: Softmax logits predicted by the model
        :param box_masks: Masks containing all the bounding boxes for each classes

        Shape:
             - ``logits``: :math:`(B \times C \times W \times H \times D)` or :math:`(B \times C \times W \times H)`
             - ``box_masks``: :math:`(B \times C \times N \times W \times H \times D)` or :math:`(B \times C \times N
               \times W \times H)` with sparse dimensions :math:`B, C, N`

            With :math:`B`, :math:`C`, :math:`N` and :math:`(W, H, D)` being respectively the batch size, the number of
            classes, the maximum number of bonding boxes and the images dimensions.

            The fact that :math:`N` is the maximum number of boxes means that each class needs to be padded to match
            the size of the class with the maximum number of bonding boxes. Reason why we use sparse matrices

        .. warning::
           This module does not check if bounding boxes are not overlapping, this matter need to be taken care during
           transformation of the bonding boxes
        """
        # Check if some boxes are present in the mask
        if box_masks.size(dim=2) == 0:
            # If no box is present, no need to compute the loss, we'll just return a loss of 0 as there is no
            # information to use for this specific loss.
            # We still need to yield a result computed from the given logits to connect the gradient graph
            # OPTIMIZATION NOTE: N-dim=0, dense-sum has 0 value to sum up
            predicted_boxes = _mask_logits(logits, box_masks)
            if self.reduction is Reduction.NONE:
                return predicted_boxes.to_dense().sum(tuple(range(3, box_masks.dim())))
            return predicted_boxes.to_dense().sum()

        # Retrieve image dimensions to support both 2D and 3D
        # This all the dimensions but the first 3, being batches, classes and number of bonding boxes
        image_dims = tuple(range(3, box_masks.dim()))

        # This loss does not care about the background class, we can drop it
        fg_indices = torch.arange(1, box_masks.size(1)).to(box_masks.device)  # Move indices to the same device as masks
        box_masks = box_masks.index_select(1, fg_indices)
        logits = logits[:, 1:, ...]

        # Get actual box surface
        # Sum over image dimensions then keep batches and classes
        # Bug https://github.com/pytorch/pytorch/issues/122711
        # box_sizes = torch.sparse.sum(box_masks, dim=image_dims)  # B x C-1 x N
        # Workaround:
        box_sizes = _torch_sparse_sum_all_dense_dim(box_masks)  # B x C-1 x N
        box_sizes = box_sizes.to_dense()

        # Get target size range (per box, outside of background)
        min_size = self.minimum * box_sizes  # B x C-1 x N
        max_size = self.maximum * box_sizes  # B x C-1 x N

        # Get predicted surface in the boxes
        #  1. Multiply logits with the boxes masks
        #  2. Sum over image dimensions
        #  3. Keep size for each batch x class
        # NOTE: This is faster than torch.einsum, even with unknown dimensions
        masked_logits = _mask_logits(logits, box_masks)  # B x C x N x Im...
        actual_sizes = torch.sparse.sum(masked_logits, dim=image_dims)  # B x C-1 x N
        actual_sizes = actual_sizes.to_dense()

        assert actual_sizes.size() == box_masks.size()[:3]

        # Penalize if predicted soft size is greater than maximum proportion or less than minimal proportion
        error = self.penalty(actual_sizes - max_size) + self.penalty(min_size - actual_sizes)
        assert error.size() == box_masks.size()[:3], 'Penalty should not change the dimensions of the tensor'

        # Reduce loss and return
        if self.reduction is Reduction.MEAN:
            return torch.mean(error / (box_sizes | torch.ones_like(box_sizes)))
        if self.reduction is Reduction.SUM:
            return torch.sum(error / (box_sizes | torch.ones_like(box_sizes)))
        if self.reduction is Reduction.ORIGINAL:
            im_shape = logits.size()[-3:]
            return torch.sum(error) / math.prod(im_shape)
        return error


class OutsideBoxEmptinessConstraintLoss(nn.Module):
    r"""
    Penalize any pixel predicted outside the target bonding boxes. "Emptiness constraint" from `H. Kervadec et al.`__.

    __ https://arxiv.org/abs/2004.06816

    .. math:: \sum_{p \in \Omega_O} s_{\theta}(p) \le 0

    With :math:`\Omega_O` being the set of pixels outside any bounding box and :math:`s_{\theta}(p)` the softmax
    background probability of a given pixel.

    .. note::
        This implementation results in exactly the same result as H. Kervadec's one, except that it is averaged over
        the classes and images in the batch instead of summed. This behavior can be obtained using
        ``OutsideBoxEmptinessConstraintLoss(..., reduction='original')``

    .. note::
        In case of training with multiple dataset and possibly unspecified classes for certain volumes, the loss need to
        know which are the specified classes for the processed volumes as absent classes must not be considered
        background. The loss must focus on the specified classes and give an error if voxels/pixels were predicted
        outside the boxes for the specified classes. Contrary to other losses in this module, this loss cannot only
        focus on provided boxes as a class that doesn't have any boxes (which means no voxel/pixel should be predicted)
        is indistinguishable from an absent class, i.e. ``0`` specified boxes. To overcome this issue, `annotation_mask`
        must be provided to the forward pass.

    :param penalty: Penalty to apply before the reduction when the error has been computed
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'```|
        ``'original'``. ``'original'`` means the original implementation of the paper: sum of all errors divided by the
        number of pixels/voxels in the input image.
    """

    def __init__(self, penalty: Callable[[torch.Tensor], torch.Tensor] = None, reduction: ReductionArg = 'original'):
        super().__init__()
        self.penalty = penalty if penalty is not None else InequalityL2Penalty()
        self.reduction = Reduction(reduction)

    def forward(self, logits: torch.Tensor, box_masks: torch.Tensor, annotation_mask: torch.Tensor = None):
        r"""
        Computes the Emptiness Constraint loss from the given logits and bonding box masks.

        :param logits: Softmax logits predicted by the model
        :param box_masks: Masks containing all the bounding boxes for each class
        :param annotation_mask: specifies which class should be considered present when computing the loss. If not
            specified, the loss will consider all classes are present and any pixel/voxel outside a box will be
            penalized

        Shape:
             - ``logits``: :math:`(B \times C \times W \times H \times D)` or :math:`(B \times C \times W \times H)`
             - ``box_masks``: :math:`(B \times C \times N \times W \times H \times D)` or :math:`(B \times C \times N
               \times W \times H)` with sparse dimensions :math:`B, C, N`
             - ``annotation_mask``: *(if provided)*  :math:`(B \times C)`

            With :math:`B`, :math:`C`, :math:`N` and :math:`(W, H, D)` being respectively the batch size, the number of
            classes, the maximum number of bonding boxes and the images dimensions.

            The fact that :math:`N` is the maximum number of bounding boxes means that each class needs to be padded to
            match the size of the class with the maximum number of bonding boxes. Reason why we use sparse matrices.
        """
        # Ensure box_masks is sparse as the loss is designed to handle sparse masks
        assert box_masks.is_sparse and box_masks.sparse_dim() == 3, "Expected a sparse box mask with sparse `B, C, N`"
        
        # Ensure annotation mask has the intended shape
        assert annotation_mask is None or annotation_mask.shape == logits.shape[:2], \
            "The annotation mask should match the batch received"

        # Retrieve image dimensions to support both 2D and 3D
        # This all the dimensions but the first 2, being batches and classes
        image_dims = tuple(range(2, len(logits.size())))

        # Compute soft size prediction outside the boxe
        #  1. Compute a mask combining all boxes for each class
        #  2. Drop the background as the loss focuses on the known classes
        #     > Required because, as opposed to other losses, we mask with the opposite of the boxes, meaning the
        #     > background won't be masked away during the process
        #  3. Invert mask to "outside mask"
        #  4. Mask the logits
        #  5. Sum them over the image dimensions to get the outside soft size
        fg_indices = torch.arange(1, box_masks.size(1)).to(box_masks.device)  # Move indices to the same device as masks
        all_boxes_mask = torch.sparse.sum(box_masks.index_select(1, fg_indices), dim=2).bool()  # Shape B x C-1 x Im...
        logits = logits[:, 1:, ...]  # Drops background
        outside_mask = ~all_boxes_mask.to_dense()
        outside = torch.sum(logits * outside_mask, dim=image_dims)
        assert outside.size() == logits.size()[:2]  # Shape: B x C-1

        # Apply penalty
        error = self.penalty(outside)
        assert error.size() == logits.size()[:2], 'penalty should not change the dimensions of the tensor'

        # We do not want any loss, thus any penalty either, for absent classes
        if annotation_mask is not None:
            error *= annotation_mask[:, 1:]  # Skip background

        # Reduce loss and return
        if self.reduction is Reduction.MEAN:
            return torch.sum(error) / (outside_mask.sum() + 1E-5)  # + perturbation to avoid Inf is sum is 0
        if self.reduction is Reduction.SUM:
            return torch.sum(error)
        if self.reduction is Reduction.ORIGINAL:
            im_shape = logits.size()[2:]  # Get shape without batch and classes
            return torch.sum(error) / math.prod(im_shape)
        return error  # reduction == 'none'


@dataclass
class LossWrapper:
    """Wraps loss values to pass through ignite's supervised_training_step without loosing computed losses"""
    tightness_prior: torch.Tensor
    box_size: torch.Tensor
    emptiness_constraint: torch.Tensor
    weighted_loss: torch.Tensor

    def __getattr__(self, item):
        return getattr(self.weighted_loss, item)


class CombinedLoss(nn.Module):
    """TODO"""
    ratios: torch.Tensor
    softmax_low_temperature: torch.Tensor
    softmax_high_temperature: torch.Tensor

    def __init__(self, reduction, ratios: Tuple[float, float, float] = None,
                 softmax_high_temperature=1E2, softmax_low_temperature=1E-1,
                 mode=Mode.MODE_3D):
        super().__init__()

        if ratios is None:
            ratios = (1., 1., 1.)

        self.log_barrier = LogBarrierPenalty()
        self.tightness_prior = BoxTightnessPriorLoss(5, penalty=self.log_barrier, reduction=reduction, mode=mode)
        self.box_size = BoxSizePriorLoss(.3, .75, penalty=self.log_barrier, reduction=reduction)
        self.emptiness_constraint = OutsideBoxEmptinessConstraintLoss(penalty=self.log_barrier, reduction=reduction)
        self.register_buffer('ratios', torch.tensor(ratios, dtype=torch.float))
        self.register_buffer('softmax_high_temperature', torch.tensor(softmax_high_temperature, dtype=torch.float))
        self.register_buffer('softmax_low_temperature', torch.tensor(softmax_low_temperature, dtype=torch.float))

    def forward(self, logits: torch.Tensor, box_masks: torch.Tensor, annotation_mask: torch.Tensor = None):
        low_temp_logits = torch.softmax(logits / self.softmax_low_temperature, dim=1)
        high_temp_logits = torch.softmax(logits / self.softmax_high_temperature, dim=1)

        tightness_prior = self.tightness_prior(low_temp_logits, box_masks)
        box_size = self.box_size(low_temp_logits, box_masks)
        emptiness_constraint = self.emptiness_constraint(high_temp_logits, box_masks, annotation_mask)

        losses = torch.stack([
            tightness_prior,
            box_size,
            emptiness_constraint,
        ])

        result = torch.sum(self.ratios * losses)

        return LossWrapper(tightness_prior, box_size, emptiness_constraint, result)

    def step(self):
        self.log_barrier.step()
