import pytest
import torch
from torch.nn import functional as F

from utils.templates import Template, UNIVERSAL_TEMPLATE, NUM_CLASSES, BACKGROUND


@pytest.fixture
def template_with_skipping():
    return Template(
        'liver',
        'skipped_class',  # skipped because not present in UNIVERSAL_TEMPLATE
        'liver_tumor',
    )


def test_skipped_indices(template_with_skipping):
    assert template_with_skipping.skipped_indices == {2}


def test_to_universal_indexing_skipped_classes(template_with_skipping):
    volume = torch.tensor([
        [3, 1, 0],
        [1, 1, 0],
        [0, 2, 2],
    ]).unsqueeze(0)

    result = template_with_skipping.to_universal_indexing(volume)

    liver, tumor = UNIVERSAL_TEMPLATE['liver'], UNIVERSAL_TEMPLATE['liver_tumor']
    torch.testing.assert_close(result, torch.tensor([
        [tumor, liver, 0],
        [liver, liver, 0],
        [0,     0,     0],
    ]).unsqueeze(0))


def test_convert_one_hot_skipped_classes(template_with_skipping):
    volume = F.one_hot(torch.tensor([
        [3, 1, 0],
        [1, 1, 0],
        [0, 2, 2],
    ], dtype=torch.long).unsqueeze(0)).permute((-1, 0, 1, 2))
    background, liver, skipped, tumor = volume
    result = template_with_skipping.convert_one_hot(volume)

    assert result.size(0) == NUM_CLASSES
    torch.testing.assert_close(result[0], background + skipped)
    torch.testing.assert_close(result[UNIVERSAL_TEMPLATE['liver']], liver)
    torch.testing.assert_close(result[UNIVERSAL_TEMPLATE['liver_tumor']], tumor)

    for i in range(1, NUM_CLASSES):
        if i not in {UNIVERSAL_TEMPLATE['liver'], UNIVERSAL_TEMPLATE['liver_tumor']}:
            torch.testing.assert_close(result[i], torch.zeros(1, 3, 3), check_dtype=False)


def test_annotation_mask_skipped_classes(template_with_skipping):
    mask = template_with_skipping.annotation_mask

    assert mask[0] == 1
    assert mask[UNIVERSAL_TEMPLATE['liver']] == 1
    assert mask[UNIVERSAL_TEMPLATE['liver_tumor']] == 1
    for i in range(1, NUM_CLASSES):
        if i not in {UNIVERSAL_TEMPLATE['liver'], UNIVERSAL_TEMPLATE['liver_tumor']}:
            assert mask[i] == 0
