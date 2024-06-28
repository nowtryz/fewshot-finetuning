import math
from itertools import zip_longest
from typing import NamedTuple, Type, Union

import numpy
import pytest
import torch
from monai.data import MetaTensor

from boundingboxes.transforms import DegradeToBoundingBoxes, TooManyBoxesError


def as_meta_tensor(tensor):
    return MetaTensor(tensor, meta={
        'filename_or_obj': 'filepath'
    })


VALID_DTYPES = {  # And number of usable bits
    numpy.int8:  7,
    numpy.uint8: 8,
    numpy.int16: 15,
    numpy.int32: 31,
    numpy.int64: 63,
    torch.int8:  7,
    torch.uint8: 8,
    torch.int16: 15,
    torch.int32: 31,
    torch.int64: 63,
}


@pytest.fixture(params=VALID_DTYPES.keys(), ids=lambda d: str(d) if isinstance(d, torch.dtype) else 'np.' + d.__name__)
def valid_dtype(request) -> Union[Type[numpy.generic], torch.dtype]:
    return request.param


@pytest.fixture
def max_boxes(valid_dtype) -> int:
    return VALID_DTYPES[valid_dtype]


# Utilities for ball volumes
class Ball(NamedTuple):
    centroid: tuple[int, int, int]
    radius: float

    def as_box(self) -> tuple[slice, slice, slice]:
        x, y, z = self.centroid
        radius = int(self.radius)
        # slice's stop can be out of bound, python does not care, however negative  values are an issue
        return (slice(max(0, x - radius), x + radius + 1),
                slice(max(0, y - radius), y + radius + 1),
                slice(max(0, z - radius), z + radius + 1))


class Class(NamedTuple):
    balls: list[Ball]
    boxes: list[tuple[slice, slice, slice]]


class TestDegradeToBoundingBoxes:
    @staticmethod
    def indices(dims):
        """
        Build tensor of indices similar to :ref:`np.indices`
        :param dims: dimensions of the tensor to index
        """
        return [
            torch.arange(0, size).view(1, 1, size).movedim(-1, dim).expand(dims)
            for dim, size in enumerate(dims)
        ]

    def test_degrade_to_bounding_boxes_ball(self):
        """
        Test the degradation of a segmentation containing balls. This test is designed to ensure the transform can handle
        multiple independent component from the same class as well as multiple classes.

        We have a class containing 2 merged balls as well as another ball somewhere else in the volume and 2 other classe
        containing one ball each. The balls are specified by there centroids and radii. We then compute the array of indices
        (c.f. numpy.indices for format information) to apply the sphere equation and fill everything inside with the class
        index.

        """
        # Build a segmentation mask matching the test case description

        dims = 20, 20, 20
        x, y, z = self.indices(dims)
        segmentation = torch.zeros((4, *dims), dtype=torch.bool)
        labels = [
            # For each class, the label is the coordinates for a sphere centroid accompanied by its radius and the
            # according boxes the transformation should produce. Those coordinates will be used in a sphere equation
            # 1. First class: two balls fused
            Class(balls=[Ball(centroid=(3, 1, 2), radius=7),
                         Ball((8, 5, 6), radius=math.sqrt(17))],
                  boxes=[(slice(13), slice(10), slice(11))]),
            # 2. Second class: two balls in different places
            Class(balls=[Ball(centroid=(4, 16, 7), radius=6),
                         Ball(centroid=(13, 5, 15), radius=4)],
                  boxes=[Ball(centroid=(4, 16, 7), radius=6).as_box(),
                         Ball(centroid=(13, 5, 15), radius=4).as_box()]),
            # 3. Third class: one single ball
            Class(balls=[Ball(centroid=(15, 13, 8), radius=math.sqrt(20))],
                  boxes=[Ball(centroid=(15, 13, 8), radius=math.sqrt(20)).as_box()]),
        ]

        for class_index, class_ in enumerate(labels):
            for (x_0, y_0, z_0), radius in class_.balls:
                segmentation[class_index + 1, (x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2 <= radius ** 2] = True

        # Transform the designed segmentation
        transform = DegradeToBoundingBoxes(dtype=torch.uint8, dynamic=False, merge_inner=False)
        result = transform(as_meta_tensor(segmentation))

        # Assertions
        assert result.shape == (4, *dims)
        torch.testing.assert_close(result[0], torch.zeros(*dims, dtype=torch.uint8))
        for class_index, class_ in enumerate(labels):
            for box_index, box in zip_longest(range(2), class_.boxes):
                expected = torch.zeros(dims, dtype=torch.uint8)
                if box is not None:
                    expected[box] = 1 << box_index
                torch.testing.assert_close(result[class_index+1] & 1 << box_index, expected)

    def test_degrade_overlapping_boxes(self):
        """
        Create a cubic segmentation split in 2 halves, the cube is halved diagonally, and both halves are assigned to
        the only class used but with an empty diagonal, resulting in 2 distinct components that should have overlapping
        bounding boxes in the middle of the volumes.
        """
        dims = 6, 6, 6
        x, y, z = self.indices(dims)
        segmentation = torch.zeros(2, *dims, dtype=torch.bool)
        segmentation[1, (x+y+z) < 5] = True  # Bottom Tetrahedron
        segmentation[1, (x+y+z) > 10] = True  # Top Tetrahedron

        # Transform the designed segmentation
        transform = DegradeToBoundingBoxes(dtype=torch.uint8, dynamic=False, merge_inner=False)
        result = transform(as_meta_tensor(segmentation))
        expected = torch.zeros(dims, dtype=torch.uint8)
        expected[:5, :5, :5] = 0b01  # Box produced by the 1st component must be 01
        expected[1:, 1:, 1:] = 0b10  # Box produced by the 2nd component must be 10
        expected[1:5, 1:5, 1:5] = 0b11  # But where the box of the 2 components are present, the result would be 11

        # Assertions
        assert result.shape == (2, *dims)
        torch.testing.assert_close(result[0], torch.zeros(*dims, dtype=torch.uint8))
        torch.testing.assert_close(result[1], expected)

    @pytest.mark.parametrize('dynamic', [True, False])
    def test_degrade_too_many_boxes(self, valid_dtype, max_boxes, dynamic):
        transform = DegradeToBoundingBoxes(dtype=valid_dtype, dynamic=dynamic, merge_inner=False)
        component_count = max_boxes + 1  # minimum number of boxes to make the binary encoding fail
        segmentation = (torch.tensor([[0, 0], [0, 1]])
                        .tile(component_count)
                        .view(2, 1, 1, 2 * component_count)
                        .expand(2, 4, 4, 2 * component_count))
        with pytest.raises(TooManyBoxesError):
            transform(as_meta_tensor(segmentation))

    def test_degrade_max_boxes(self, valid_dtype, max_boxes):
        """
        Check that all the valid types can be used for degradation with the maximum number of boxes without failing
        """
        transform = DegradeToBoundingBoxes(dtype=valid_dtype, dynamic=False, merge_inner=False)
        segmentation = (torch.tensor([[0, 0], [0, 1]])
                        .tile(max_boxes)
                        .view(2, 1, 1, 2 * max_boxes)
                        .expand(2, 4, 4, 2 * max_boxes))
        transform(as_meta_tensor(segmentation))

    @pytest.mark.parametrize('valid_dtype', (t for t in VALID_DTYPES.keys() if isinstance(t, torch.dtype)),
                             indirect=True, ids=str)
    def test_degrade_dynamic(self, valid_dtype, max_boxes):
        """Check that the smallest dtype supporting the given number of boxes is selected"""
        transform = DegradeToBoundingBoxes(dtype=numpy.int64, dynamic=True, merge_inner=False)
        segmentation = (torch.tensor([[0, 0], [0, 1]])
                        .tile(max_boxes)
                        .view(2, 1, 1, 2 * max_boxes)
                        .expand(2, 4, 4, 2 * max_boxes))
        result = transform(as_meta_tensor(segmentation))
        assert result.dtype == valid_dtype

    def test_degrade_to_bounding_boxes_empty_classes(self):
        degrade = DegradeToBoundingBoxes(has_background=False, dynamic=False, merge_inner=False)
        label = torch.zeros(2, 10, 10, 10, dtype=torch.bool)
        label[1, 3:9, 3:9, 3:9] = True
        label_bb = degrade(as_meta_tensor(label))
        assert label_bb.size() == label.size()
        assert torch.equal(label_bb[1, 3:9, 3:9, 3:9], torch.ones(6, 6, 6)) is True
        assert torch.equal(label_bb[0], torch.zeros(10, 10, 10)) is True

    def test_invalid_torch_dtype(self):
        with pytest.raises(ValueError):
            DegradeToBoundingBoxes(dtype=torch.quint8)

    @pytest.mark.parametrize('dtype', [
        # because torch only supports uint8 and not bigger unsigned ints at the moment. If so, a dynamic detection of
        # available dtypes could have been used.
        numpy.uint16, numpy.uint32, numpy.uint64,
        # Not integers
        *[dtype for cls in numpy.inexact.__subclasses__() for dtype in cls.__subclasses__()]
    ])
    def test_invalid_numpy_dtype(self, dtype):
        with pytest.raises(ValueError):
            DegradeToBoundingBoxes(dtype=dtype)

    def test_merge_inner(self):
        degrade = DegradeToBoundingBoxes(merge_inner=True, dynamic=False, has_background=False, dtype=numpy.uint8)
        dims = 10, 10, 10
        x, y, z = self.indices(dims)
        label = torch.zeros(1, *dims, dtype=torch.bool)
        label[0, (x + y + z) <= 8] = True  # Tetrahedron
        label[0, -3, -3, -3] = True
        label[0,  0, -2, -2] = True
        label[0, -2,  0, -2] = True
        label[0, -2, -2,  0] = True
        label[0,  5,  5,  5] = True
        label[0, -1, -1, -1] = True

        result = degrade(as_meta_tensor(label))
        expected = torch.zeros(1, *dims, dtype=torch.uint8)
        expected[0, :-1, :-1, :-1] = 1 << 0
        expected[0,  -1,  -1,  -1] = 1 << 1
        torch.testing.assert_close(result, expected)

    def test_has_single_component(self):
        degrade = DegradeToBoundingBoxes(has_background=False, dynamic=False, merge_inner=False, mono_components=[True])
        dims = 10, 10, 10

        label = torch.zeros(1, *dims, dtype=torch.bool)
        label[0, -2, -2, -2] = True
        label[0,  1, -2, -2] = True
        label[0, -2,  1, -2] = True
        label[0, -2, -2,  1] = True
        label[0,  1,  1,  1] = True
        label[0, -2,  1,  1] = True
        label[0,  1, -2,  1] = True
        label[0,  1,  1, -2] = True
        label[0,  4,  4,  4] = True

        result = degrade(as_meta_tensor(label))
        expected = torch.zeros(1, *dims, dtype=torch.int)
        expected[0, 1:-1, 1:-1, 1:-1] = 1 << 0
        torch.testing.assert_close(result, expected, check_dtype=False)


class TestBoundingBoxesToOneHot:
    pass
