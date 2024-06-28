from pathlib import Path

import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import LoadImage
from torch.nn import functional as F  # noqa

from boundingboxes.transforms import DegradeToBoundingBoxes, BoundingBoxesToOneHot, OriginalDimToUniversalDim, \
    CategoricalToOneHotDynamic
from utils.templates import BTCV_TEMPLATE, NUM_CLASSES


@pytest.fixture
def data_directory(request):
    return Path(__file__).parents[2]


@pytest.fixture
def atlas_label_1(data_directory) -> MetaTensor:
    path = data_directory / 'data/01_Multi-Atlas_Labeling/label/label0001.nii.gz'

    if not path.exists():
        pytest.xfail(f'Path {path} does not exist, this test requires access to the data directory')

    loader = LoadImage(image_only=True)
    return loader(path)


def test_degrade_to_bounding_boxes(atlas_label_1):
    label = atlas_label_1
    degrade = DegradeToBoundingBoxes()
    box_to_one_hot = BoundingBoxesToOneHot(sparse=True)

    label = F.one_hot(label.long()).permute((-1, 0, 1, 2))
    label_bb = degrade(label)

    assert label_bb.size() == label.size()

    label_bb = label_bb[..., :96, :96, :96]
    bb = box_to_one_hot(label_bb)
    c, h, w, d = label_bb.size()

    assert bb.is_sparse
    assert bb.dense_dim() == 3
    assert bb.size() == (c, len(torch.unique(label_bb)) - 1, h, w, d)


def test_original_dim_to_universal_dim(atlas_label_1):
    transform = OriginalDimToUniversalDim(template_key="template", bb_key="label")
    template = BTCV_TEMPLATE
    label = atlas_label_1
    label = F.one_hot(label.long(), template.num_classes).permute((-1, 0, 1, 2))

    data = {
        'label': label,
        'template': template
    }

    result = transform(data)

    assert result['label'].shape == (NUM_CLASSES, *label.shape[1:])
    assert not result['label'][template.annotation_mask == 0].any()


def test_categorical_to_one_hot_dynamic(atlas_label_1):
    transform = CategoricalToOneHotDynamic(label_key="label", template_key="template")
    template = BTCV_TEMPLATE
    label = atlas_label_1[None, ...]

    data = {
        'label': label,
        'template': template
    }

    result = transform(data)

    assert result['label'].shape == (BTCV_TEMPLATE.num_classes, *label.shape[1:])
