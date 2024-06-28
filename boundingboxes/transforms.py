from itertools import repeat, count, chain
from typing import Mapping, Hashable, Dict, Callable, Iterable, Iterator, cast, Type, Container

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, apply_transform
from scipy import ndimage
from torch.nn import functional as F  # noqa

from utils.decorators import wrap_ndarray_in_tensor, auto_repr, preserves_meta
from utils.templates import Template


class TooManyBoxesError(RuntimeError):
    pass


@auto_repr
class DegradeToBoundingBoxes:
    """
    Transforms a 3D segmentation ground truth to 3D bounding boxes. For each class, the channel will contain a different
    value for each bonding box and it must be extracted later.

    .. note:
        Because the bonding boxes will still nee TODO

    :param has_background: Specifies if the volume provided includes a background at index ``0``, it this case it will
        be skipped
    :param dtype: The dtype in which the the boxes will be encoded. This dtype it an integer type with which each bit
        will be assigned to a detected component. Volumes requiring a lot of boxes per class will need a dtype with a
        bigger byte size. For valid dtypes, refer to :ref:`VALID_DTYPES`.
    :param dynamic: When `dynamic` is set to `True`, the transform will use the smallest dtype possible to encode the
        bounding box. The resulting tensors may not have the same dtype but will be properly handled by
        `BoundingBoxesToOneHot` later on.
    :param merge_inner: If ``True``, all boxes that are fully included in a bigger one will be skipped.
    :param mono_components: A tuple specifying for each class if it contains only one component or may contain
        multiple components. If the class should only contain one component, all detected components will be
        conveniently merge together to avoid having multiple small components.
    """
    MEASUREMENT_STRUCTURE = ndimage.generate_binary_structure(rank=3, connectivity=3)
    # Torch only supports uint8 for unsigned integer, otherwise we need to use its signed version
    # see https://pytorch.org/docs/stable/tensor_attributes.html
    # see https://github.com/pytorch/pytorch/issues/58734
    VALID_DTYPES = np.int8, np.uint8, np.int16, np.int32, np.int64,

    def __init__(self, has_background=True, dtype: Type[np.integer] | torch.dtype = np.int64, dynamic=True,
                 merge_inner=True, mono_components: tuple[bool, ...] = None):
        self.has_background = has_background
        self.dynamic = dynamic
        self.merge_inner = merge_inner
        self.mono_components = mono_components
        self.dtype = self.ensure_numpy_dtype(dtype)
        self.max_boxes = self.max_boxes_for_dtype(self.dtype)

    @classmethod
    def ensure_numpy_dtype(cls, dtype: Type[np.integer] | torch.dtype):
        if isinstance(dtype, torch.dtype):
            dtype_name = str(dtype).removeprefix('torch.')
            if not hasattr(np, dtype_name):
                raise ValueError(f'Received a torch dtype ({dtype}) that cannot be converting to its numpy equivalent')
            dtype = getattr(np, dtype_name)

        if dtype not in cls.VALID_DTYPES:
            raise ValueError(f"{cls} only supports {', '.join(str(dtype) for dtype in cls.VALID_DTYPES)} data types")

        return dtype

    @classmethod
    def max_boxes_for_dtype(cls, dtype: Type[np.integer]):
        max_boxes = np.dtype(dtype).itemsize * 8
        if not issubclass(dtype, np.unsignedinteger):
            max_boxes -= 1  # Most significant bit is used for the sign, altering the algorithm if present
        return max_boxes

    def _get_dtype(self, box_count: int):
        """Compute the smallest possible dtype for the given number of boxes when dynamic mode is enabled."""
        if not self.dynamic:
            return self.dtype

        for dtype in self.VALID_DTYPES:
            # 1. Do not take a dtype bigger than the specified one
            #    -> however this check is not needed as a TooManyBoxesError would have been raised if the next case
            #       would not cover the specified number of boxes
            # 2. Take the first dtype with enough space for the specified number of boxes
            if self.max_boxes_for_dtype(dtype) >= box_count:
                return dtype

    @preserves_meta(key='volume')
    def __call__(
            self, volume: MetaTensor, mono_components: tuple[bool, ...] = None, skip_indices: Container[int] = None
    ) -> torch.Tensor:
        """
        Use connected components to compute 3D bounding boxes from the ground truth then store in a tensor of shape
        ``Classes x Dims...`` with a different number for each component.

        C x W x H x D -> C x W x H x D

        :param volume: The segmentation to degrade to bounding boxes
        :param mono_components: A tuple specifying for each class if it contains only one component or may contain
            multiple components. If the class should only contain one component, all detected components will be
            conveniently merge together to avoid having multiple small components.
        :param: skip_indices: A set (preferred) or other container containing indices to skip during degradation. For
            these indices, the transform won't compute the bounding boxes for the corresponding class and yield an empty
            mask instead.
        """
        # TECHNICAL NOTE: when dynamic mode is enabled, the dtype may not be consistent across all tensors gathered from
        #                 _3d_segmentation_to_bounding_boxes but torch.stack will take the biggest dtype.
        classes = torch.unbind(volume, dim=0)
        mono_components = mono_components or self.mono_components or (False,) * len(classes)
        skip_mask = (i in skip_indices for i in range(len(classes))) if skip_indices else repeat(False)

        if self.has_background:
            next(skip_mask)  # skip first element of the skip mask
            skip_mask = chain([True], skip_mask)  # yield True to skip background then yield from skip mask

        return torch.stack([
            self._3d_segmentation_to_bounding_boxes(mask, idx, volume.meta.get('filename_or_obj'), has_single_component)
            if not skip else torch.zeros_like(mask, dtype=torch.bool)  # keep smallest dtype to let torch.stack choose
            for idx, mask, has_single_component, skip in zip(count(), classes, mono_components, skip_mask)
        ])

    @wrap_ndarray_in_tensor
    def _3d_segmentation_to_bounding_boxes(
            self,
            segmentation_mask: np.ndarray,
            class_idx: int,
            filename: str,
            has_single_component: bool
    ) -> np.ndarray:
        """W x H x D -> W x H x D"""
        if not np.any(segmentation_mask):
            return np.zeros(segmentation_mask.shape)

        # We compute bounding boxes, hence interior holes do not matter but filling them helps to only produce one box
        # per object, even if the segmentation contains small gaps that would have produces multiples boxes for the same
        # object.
        segmentation_mask = ndimage.binary_fill_holes(segmentation_mask, structure=self.MEASUREMENT_STRUCTURE)

        # Run scipy nd-image's connected components algorithm
        labels, num_features = ndimage.label(segmentation_mask, structure=self.MEASUREMENT_STRUCTURE)
        objects: list[tuple[slice, slice, slice]] = ndimage.find_objects(labels)  # noqa
        del labels  # Free unused arrays early

        if has_single_component:
            objects = [cast(tuple[slice, slice, slice], tuple(
                slice(
                    min(box[dim].start for box in objects),
                    max(box[dim].stop for box in objects),
                ) for dim in range(3)
            ))]
            num_features = 1
        elif self.merge_inner:
            objects = list(self._filter_inner_boxes(objects))
            num_features = len(objects)

        if num_features > self.max_boxes:
            msg = (
                f'Computed {num_features} individual bounding boxes for class {class_idx} while degrading {filename}, '
                f'but the dtype {self.dtype} can only support the binary-encoding of {self.max_boxes} boxes.'
            )
            bigger_types = (str(dtype) for dtype in self.VALID_DTYPES[self.VALID_DTYPES.index(self.dtype):])
            if bigger_types:
                msg += f"Consider using dtype with more available bits ({', '.join(bigger_types)})"
            raise TooManyBoxesError(msg)

        result = np.zeros(segmentation_mask.shape, dtype=self._get_dtype(num_features))
        for component_index, slices in enumerate(objects):
            # Bitwise OR with a different bit for each box
            result[slices] |= 0b1 << component_index

        del objects  # Free unused arrays early
        return result

    @staticmethod
    def _filter_inner_boxes(slices: Iterable[tuple[slice, slice, slice]]) -> Iterator[tuple[slice, slice, slice]]:
        slices: list[tuple[slice, slice, slice] | None] = list(slices)
        for box_index, box in enumerate(slices):
            for candidate in slices:
                if candidate is None or candidate is box:
                    continue
                if all(candidate[i].start <= box[i].start and candidate[i].stop >= box[i].stop for i in range(3)):
                    slices[box_index] = None
                    break
            else:
                yield box


@auto_repr
class BoundingBoxesToOneHot:
    r"""
    Takes an image provided by `DegradeToBoundingBoxes` (shape :math:`C \times W \times H \times D`), after
    some eventual additional transformations and convert it to the shape expected by bonding box loss functions
    (shape :math:`C \times N \times W \times H \times D`).

    :param sparse: Convert output tensor as sparse to save memory
    """
    def __init__(self, sparse=False):
        self.sparse = sparse

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume_shape = volume.size()
        assert len(volume_shape) == 4

        boxes_count = volume.max().item().bit_length()
        bb_one_hot = torch.zeros(volume_shape[0], boxes_count, *volume_shape[1:], dtype=torch.bool)

        for box_index in range(boxes_count):
            bb_one_hot[:, box_index, ...] = volume & 0b1 << box_index

        # Some boxes may have disappeared during the transformation, e.g. a RandomCrop, remove empty masks
        # We sum over all dimensions except 1 to get the indices of present boxes
        present_bb = torch.any(bb_one_hot, dim=tuple(filter(lambda i: i != 1, range(bb_one_hot.dim()))))
        bb_one_hot = bb_one_hot[:, present_bb]

        if self.sparse:
            return bb_one_hot.to_sparse(2)  # sparse_pad_sequence can then be used to build the batch

        return bb_one_hot


# ===================
# Dictionary versions
# ===================


@auto_repr
class AsDictionaryTransform(MapTransform):
    def __init__(self, transform: Callable, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.transform = transform

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:  # noqa
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = apply_transform(self.transform, data[key])
        return data


@auto_repr
class DegradeToBoundingBoxesD(MapTransform):
    def __init__(self, keys: KeysCollection, template_key: Hashable = 'template', *args,
                 allow_missing_keys: bool = False, **kwargs):
        super().__init__(keys, allow_missing_keys)
        self.transform = DegradeToBoundingBoxes(*args, **kwargs)
        self.template_key = template_key

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:  # noqa
        data = dict(data)
        template: Template = data[self.template_key]
        for key in self.key_iterator(data):
            params = (data[key], template.mono_components, template.skipped_indices)
            data[key] = apply_transform(self.transform, params, unpack_items=True)
        return data


@auto_repr
class OriginalDimToUniversalDim:
    def __init__(self, template_key: Hashable = 'template', bb_key: Hashable = 'label'):
        self.template_key = template_key
        self.bb_key = bb_key

    def __call__(self, data):
        template: Template = data[self.template_key]
        bb_label = data[self.bb_key]

        # Use advanced indexing to convert from dataset's one-hot to universal one-hot
        bb_label = template.convert_one_hot(bb_label)
        data[self.bb_key] = bb_label

        return data


@auto_repr
class CategoricalToOneHotDynamic:
    def __init__(self, label_key: Hashable = 'label', template_key: Hashable = 'template'):
        self.template_key = template_key
        self.label_key = label_key

    def __call__(self, data):
        template: Template = data[self.template_key]
        y = data[self.label_key]

        if len(y.size()) == 4 and y.size(0) == 1:
            y = y.squeeze(0)  # Remove dim created by AddChannel

        # one hot encoding
        y = torch.nn.functional.one_hot(
            y.to(torch.long),
            num_classes=template.num_classes
        ).permute(-1, 0, 1, 2).bool()

        data[self.label_key] = y
        return data
