import warnings
from typing import Mapping, Hashable, Dict, Callable

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, apply_transform
from scipy import ndimage
from torch.nn import functional as F  # noqa

from utils.decorators import wrap_ndarray_in_tensor, auto_repr, preserves_meta
from utils.templates import Template


class BoundingBoxWaring(UserWarning):
    """
    Base class for warnings regarding the computation of bounding boxes
    :param filename: Label file containing the segmentation for which the warning is issued
    :param class_index: Index of the class from the label for which the warning is issued
    """
    def __init__(self, filename, class_index):
        self.filename = filename
        self.class_index = class_index


class BoundingBoxDiscardedWarning(BoundingBoxWaring):
    """
    Warning issued when a segmentation contains multiple components (connected components) for the same class that are
    overlapping but with one being the subset of the other.
    In this case, the smaller box is discarded but this may comme from an error in the input label.
    """
    def __str__(self):
        return ('Bounding box discarded as included in a bigger one for class. '
                f'(In {self.filename} at class index {self.class_index}.')


class BoundingBoxOverlapWarning(BoundingBoxWaring):
    """
    Warning issued when multiple bounding boxes for different connected components of the same class are overlapping and
    information is lost because only one box can be stored per class for each voxel. The data structured chosen enables
    it to be passed through 3D transforms as if it was a normal multiple-channeled 3D Tensor.
    """
    def __str__(self):
        return ('Mask will contain overlapping bounding boxes. '
                f'(In {self.filename} at class index {self.class_index}.')


@auto_repr
class DegradeToBoundingBoxes:
    """
    Transforms a 3D segmentation ground truth to 3D bounding boxes. For each class, the channel will contain a different
    value for each bonding box and it must be extracted later.

    .. note:
        Because the bonding boxes will still nee TODO
    """
    def __init__(self, has_background=True):
        self.has_background = has_background

    @preserves_meta(key='volume')
    def __call__(self, volume: MetaTensor) -> torch.Tensor:
        """
        Use connected components to compute 3D bounding boxes from the ground truth then store in a tensor of shape
        ``Classes x Dims...`` with a different number for each component.

        C x W x H x D -> C x W x H x D
        """
        # TODO return actual coords as well separately on top of the mask
        classes = torch.unbind(volume, dim=0)

        if self.has_background:
            return torch.stack([torch.zeros_like(classes[0])] + [
                self._3d_segmentation_to_bounding_boxes(mask, idx + 1, volume.meta.get('filename_or_obj'))
                for idx, mask in enumerate(classes[1:])
            ])

        return torch.stack([
            self._3d_segmentation_to_bounding_boxes(mask, idx, volume.meta.get('filename_or_obj'))
            for idx, mask in enumerate(classes)
        ])

    @staticmethod
    @wrap_ndarray_in_tensor
    def _3d_segmentation_to_bounding_boxes(segmentation_mask: np.ndarray, class_idx: int, filename: str) -> np.ndarray:
        """W x H x D -> W x H x D"""
        if not np.any(segmentation_mask):
            return np.zeros(segmentation_mask.shape)

        # Run scipy nd-image's connected components algorithm
        labels, num_features = ndimage.label(segmentation_mask)
        objects = ndimage.find_objects(labels)
        result = np.zeros(segmentation_mask.shape, dtype=np.integer)

        for component, slices in enumerate(objects):
            # Only check overlapping components when python is not in optimized mode (-O flag)
            if np.any(result[slices]):
                # Overlapping boxes detected, trying to merge them if one is inside the other
                u = np.unique(result[slices])

                if u.size == 1:
                    # Discarding current boxe as included in a bigger box
                    warnings.warn(BoundingBoxDiscardedWarning(filename, class_idx))
                    continue

                if u.size == 2 and u[0] == 0:
                    # There is only one box inside the current one, let's check if the current one is bigger than the
                    # previous one
                    previous_component = u[1] - 1
                    previous_slices = objects[previous_component]
                    if np.unique(result[previous_slices]).size == 1:
                        # Current box is bigger and fully includes the previous one, filling current box with the index
                        # from the previous component
                        result[slices] = previous_component
                        warnings.warn(BoundingBoxDiscardedWarning(filename, class_idx))
                        continue

                warnings.warn(BoundingBoxOverlapWarning(filename, class_idx))

            result[slices] = component + 1

        # Free unused arrays early
        del labels
        del objects
        return result


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
        assert len(volume.size()) == 4
        bb_one_hot: torch.Tensor = F.one_hot(volume.long()).bool()

        # A background will naturally be generated where the is all 0s, it needs to be removed as the background is
        # where the is no bounding box on any of the classes. Setting it to 0 to keep the same class dimension.
        bb_one_hot[..., 0] = 0

        # New box (one hot) dimension is at the end, move it to the second position
        bb_one_hot = torch.permute(bb_one_hot, dims=(0, -1, 1, 2, 3))

        # Some boxes may have disappeared during the transformation, e.g. a RandomCrop, remove empty masks
        # NOTE: tuple dim is not supported yet in pytorch 1.13 (requires >=2.2), flattening 3 dimensions instead
        present_bb = torch.flatten(bb_one_hot, start_dim=-3).any(dim=-1).any(dim=0)
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
