"""PYTEST_DONT_REWRITE"""
import torch

from boundingboxes.transforms import DegradeToBoundingBoxes


def test_degrade_to_bounding_boxes_empty_classes():
    degrade = DegradeToBoundingBoxes()
    label = torch.zeros(2, 10, 10, 10, dtype=torch.bool)
    label[1, 3:9, 3:9, 3:9] = True
    label_bb = degrade(label)
    assert label_bb.size() == label.size()
    assert torch.all(label_bb[1, 3:9, 3:9, 3:9] == 1)  # noqa
    assert torch.all(label_bb[0] == 0)  # noqa