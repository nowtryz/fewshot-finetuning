from collections import Counter, defaultdict
from itertools import repeat, count, chain
from typing import Mapping, Hashable, Dict, Callable, Iterable, Iterator, cast, Container

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, apply_transform
from monai.utils import ensure_tuple, TransformBackends
from scipy import ndimage
from torch.nn import functional as F  # noqa

from utils.decorators import auto_repr
from utils.templates import Template


try:
    from itertools import batched
except ImportError:
    from itertools import islice

    def batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        # from docs.python.org
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


@auto_repr
class DegradeToBoundingBoxes:
    """
    Transforms a 3D segmentation ground truth to 3D bounding boxes. For each class, the channel will contain a different
    value for each bonding box and it must be extracted later.

    .. note:
        Because the bonding boxes will still nee TODO

    .. warning::
        This transformation may produce `torch.int64` tensors which would raise an error if saved in a Nifti format
        (``.nii.gz`` or ``.nii``). If you plan to use :ref:`monai.transforms.SaveImage`, consider using explicitly the
        writer :ref:`.data.BoundingBoxWriter` as it will allow to store the volumes in different dtypes according to the
        one produce by this transform. You should also consider that many monai transforms will convert tensors to
        `torch.float32` by default, which would break the binary encoding used in this transform, consider specifying
        the ``dtype=None`` parameter for those transforms.

    :param has_background: Specifies if the volume provided includes a background at index ``0``, it this case it will
        be skipped
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

    def __init__(self, has_background=True, merge_inner=True, mono_components: tuple[bool, ...] = None):
        self.has_background = has_background
        self.merge_inner = merge_inner
        self.mono_components = mono_components

    def __call__(
            self, volume: MetaTensor, mono_components: tuple[bool, ...] = None, skip_indices: Container[int] = None
    ) -> tuple[MetaTensor, torch.Tensor]:
        """
        Use connected components to compute 3D bounding boxes from the ground truth then store in a tensor of shape
        ``Classes x Dims...`` with a different number for each component. TODO update

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
        #                 _3d_segmentation_to_bounding_boxes but torch.stack will take the biggest dtype. TODO update
        volume = volume.squeeze(0)  # squeeze class dim if present
        num_classes = torch.max(volume).int().item()
        mono_components = mono_components or self.mono_components or (False,) * num_classes
        skip_mask = (i in skip_indices for i in range(num_classes)) if skip_indices else repeat(False)

        if self.has_background:
            next(skip_mask)  # skip first element of the skip mask
            skip_mask = chain([True], skip_mask)  # yield True to skip background then yield from skip mask

        boxes_list = [
            self._3d_segmentation_to_bounding_boxes(volume == idx, has_single_component)
            if not skip else None
            for idx, has_single_component, skip in zip(count(), mono_components, skip_mask)
        ]
        labels = list(chain.from_iterable(
            repeat(idx, times=len(boxes))
            for idx, boxes in enumerate(boxes_list)
            if boxes is not None
        ))
        boxes_tensor = torch.cat([
            torch.tensor(boxes, dtype=torch.long)
            for boxes in boxes_list
            if boxes is not None
        ]) if labels else torch.zeros(0, volume.dim() * 2, dtype=torch.long)

        # TODO test that tensor has shape N x 4 or N x 6 in unit tests (when empty and not empty)
        return MetaTensor(boxes_tensor, meta=volume.meta), torch.tensor(labels)

    def _3d_segmentation_to_bounding_boxes(
            self,
            segmentation_mask: torch.Tensor,
            has_single_component: bool
    ) -> list[list[int]] | None:
        """W x H x D -> W x H x D"""
        segmentation_mask = segmentation_mask.numpy()

        if not np.any(segmentation_mask):
            return None

        # We compute bounding boxes, hence interior holes do not matter but filling them helps to only produce one box
        # per object, even if the segmentation contains small gaps that would have produces multiples boxes for the same
        # object.
        segmentation_mask = ndimage.binary_fill_holes(segmentation_mask, structure=self.MEASUREMENT_STRUCTURE)

        # Run scipy nd-image's connected components algorithm
        labels, num_features = ndimage.label(segmentation_mask, structure=self.MEASUREMENT_STRUCTURE)
        objects: list[tuple[slice, slice, slice]] = ndimage.find_objects(labels)  # noqa
        del labels  # Free unused arrays early

        if has_single_component:
            objects = self._merge_all_boxes(objects)
        elif self.merge_inner:
            objects = list(self._filter_inner_boxes(objects))

        result = [
            [index for axis_slice in component for index in (axis_slice.start, axis_slice.stop)]
            for component in objects
        ]

        del objects  # Free unused arrays early
        return result

    @staticmethod
    def _merge_all_boxes(objects: list[tuple[slice, slice, slice]]) -> list[tuple[slice, slice, slice]]:
        return [cast(tuple[slice, slice, slice], tuple(
            slice(
                min(box[dim].start for box in objects),
                max(box[dim].stop for box in objects),
            ) for dim in range(3)
        ))]

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
    (shape :math:`C \times N \times W \times H \times D`). TODO update

    :param sparse: Convert output tensor as sparse to save memory
    """

    def __init__(self, sparse=False, num_classes=None):
        self.sparse = sparse
        self.num_classes = num_classes

    @staticmethod
    def _iter_boxes(boxes, labels):
        counter = defaultdict(int)

    def __call__(
            self,
            boxes: torch.Tensor,
            labels: list[int],
            volume_shape: torch.Size,
            num_classes=None,
    ) -> torch.Tensor:
        assert boxes.size(1) == 6
        assert boxes.size(0) == labels.size(0)

        num_classes = num_classes or self.num_classes or max(labels)
        boxes_dim = Counter(labels).most_common(1)[0][1] if boxes.size(0) > 0 else 0
        bb_one_hot = torch.zeros(num_classes, boxes_dim, *volume_shape, dtype=torch.bool)

        if boxes.size(0) > 0:
            counter = defaultdict(int)
            for box, label in zip(boxes, labels):
                slices = [slice(start, stop) for start, stop in batched(box, 2)]
                bb_one_hot[label, counter[label], *slices] = True
                counter[label] += 1

        # TODO is is still necessary?
        # Some boxes may have disappeared during the transformation, e.g. a RandomCrop, remove empty masks
        # We sum over all dimensions except 1 to get the indices of present boxes
        # present_bb = torch.any(bb_one_hot, dim=tuple(filter(lambda i: i != 1, range(bb_one_hot.dim()))))
        # bb_one_hot = bb_one_hot[:, present_bb]

        if self.sparse:
            return bb_one_hot.to_sparse(2)  # sparse_pad_sequence can then be used to build the batch

        return bb_one_hot


# ===================
# Dictionary versions
# ===================

@auto_repr
class DegradeToBoundingBoxesD(MapTransform):
    backend = TransformBackends.TORCH

    def __init__(self,
                 keys: KeysCollection,
                 box_keys: KeysCollection,
                 box_labels_keys: KeysCollection,
                 template_key: Hashable = 'template',
                 *args,
                 allow_missing_keys: bool = False,
                 **kwargs):
        super().__init__(keys, allow_missing_keys)
        self.transform = DegradeToBoundingBoxes(*args, **kwargs)
        self.template_key = template_key
        self.box_keys = ensure_tuple(box_keys)
        self.box_labels_keys = ensure_tuple(box_labels_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:  # noqa
        data = dict(data)
        template: Template = data[self.template_key]
        for key, box_key, box_label_key in self.key_iterator(data, self.box_keys, self.box_labels_keys):
            data[box_key], data[box_label_key] = apply_transform(
                self.transform,
                data=(data[key], template.mono_components, template.skipped_indices),
                unpack_items=True,
                map_items=False
            )
        return data


@auto_repr
class BoundingBoxesToOneHotD(MapTransform):
    backend = TransformBackends.TORCH

    def __init__(
            self,
            boxes_keys: KeysCollection,
            labels_keys: KeysCollection,
            box_ref_image_keys: KeysCollection,
            num_classes: int | None,
            sparse=True
    ):
        super().__init__(boxes_keys)
        self.transform = BoundingBoxesToOneHot(sparse)
        self.labels_keys = ensure_tuple(labels_keys)
        self.box_ref_image_keys = ensure_tuple(box_ref_image_keys)
        self.labels_keys = ensure_tuple(labels_keys)
        self.num_classes = num_classes

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:  # noqa
        d = dict(data)

        for boxes_key, labels_key, ref_image_key in self.key_iterator(data, self.labels_keys, self.box_ref_image_keys):
            boxes = d[boxes_key]
            box_labels = d[labels_key]
            spatial_shape = d[ref_image_key].shape[1:]
            d[boxes_key] = apply_transform(
                self.transform,
                data=(boxes, box_labels, spatial_shape, self.num_classes),
                unpack_items=True,
                map_items=False
            )

        return d
