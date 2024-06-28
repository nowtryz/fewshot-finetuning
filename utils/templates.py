# Unified template
from collections import UserDict
from typing import Dict

import numpy as np
import torch


def _labels_to_dict(*labels: str) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(labels, start=1)}


UNIVERSAL_TEMPLATE = _labels_to_dict(
    'spleen',
    'rkidney', 'lkidney',
    'gall',
    'esophagus',
    'liver',
    'stomach',
    'aorta',
    'postcava',
    'psv',  # portal vein and splenic vein  # ?
    'pancreas',
    'radrenal',
    'ladrenal',
    'duodenum',
    'bladder',
    'prost_ut',
    'liver_tumor', 'kidney_tumor', 'kidney_cyst',
    'celiac_truck',
    'rlung', 'llung',
    # 'bone',
    'brain',
    'lung_tumor',
    'pancreas_tumor',
    # 'hepatic_vessels',
    'hepatic_tumor',
    'colon_tumor'
)

NUM_CLASSES = np.max(np.fromiter(UNIVERSAL_TEMPLATE.values(), dtype=int)) + 1
BACKGROUND = 'bg'
MONO_COMPONENT_CLASSES = {
    # Organs
    'spleen', 'liver', 'stomach', 'pancreas', 'bladder', 'gall', 'brain', 'prost_ut',
    'rkidney', 'lkidney', 'rlung', 'llung', 'radrenal', 'ladrenal',
    # Not organs but similar
    'duodenum', 'esophagus',
    # Arteries:
    'celiac_truck',
    # What about psv?

}
"""
Classes that contains only one component.
These classes can be converted to a single box when degraded to bounding boxes.
"""


class Template(UserDict):
    """
    Given a list of labels for each of the classes present in the dataset, this object allow to match classes between
    dataset using the `UNIVERSAL_TEMPLATE` containing all the classes used during training. If a class is present in the
    given list of labels but not in the universal label, it will be skipped and not used when converting tensors to the
    unified template.
    """
    def __init__(self, *class_labels: str):
        super().__init__(_labels_to_dict(*class_labels))
        self.num_classes = len(class_labels) + 1
        self.labels = (BACKGROUND, *class_labels)
        self.skipped_labels = {label for label in class_labels if label not in UNIVERSAL_TEMPLATE}
        self.skipped_indices = {self[label] for label in self.skipped_labels}

        # Build indices: allow converting original indices to universal indices with advanced indexing
        # -1 in the reversed index is used to yield an empty class, as the background will be filled with some values
        self._mapping_index = np.zeros(self.num_classes)  # used to convert categorical labels
        self._reverse_index = -np.ones(NUM_CLASSES)  # used to convert one-matrices
        self._reverse_index[0] = 0  # Match backgrounds
        for category, original_idx in self.items():
            universal_idx = UNIVERSAL_TEMPLATE.get(category)
            if universal_idx is not None:
                self._mapping_index[original_idx] = universal_idx
                self._reverse_index[universal_idx] = original_idx
            else:
                # NOTE: Skipped classes must yield a 0 -> treated as background
                #       => self._mapping_index[skipped_class] = 0
                self._mapping_index[original_idx] = 0

        self._mapping_index_torch = torch.tensor(self._mapping_index)

    def to_universal_indexing(self, volume):
        """
        Use advanced indexing to convert categorical values from the dataset to their universal equivalent
        """
        if isinstance(volume, torch.Tensor):
            dtype = volume.dtype
            return self._mapping_index_torch[volume.long()].to(dtype)
        return self._mapping_index[volume]

    def convert_one_hot(self, volume):
        """
        Convert one-hot volume in the original indices to its one-hot equivalent in the universal indices
        """
        pad = torch.zeros_like(volume[[0]])  # Create a blank class with the volume dimension
        volume = torch.cat([volume, pad])  # Append pad as last element, so that reversed index `-1` yields the padding
        result = volume[self._reverse_index]

        # Skipped classes must yield the one-hot vector of the background. So we test if the template contains skipped
        # classes, if so, we add them to the background
        if self.skipped_indices:
            result[0] |= volume[list(self.skipped_indices)].any(0)

        return result

    @property
    def annotation_mask(self):
        """
        A vector of the size of the universal template where 1 means the class is present in the dataset and 0 means the
        class is absent
        """
        return np.array(self._reverse_index != -1).astype(int)

    @property
    def mono_components(self):
        """
        A boolean vector of boolean values indicating whether each class has only one component per class or can be a
        collection of multiple disconnected components like tumorous regions
        """
        return tuple(label in MONO_COMPONENT_CLASSES for label in self.labels)


# 01. BTCV (Multi-atlas)
BTCV_TEMPLATE = Template(
    'spleen',  # 1
    'rkidney',  # 2
    'lkidney',  # 3
    'gall',  # 4
    'esophagus',  # 5
    'liver',  # 6
    'stomach',  # 7
    'aorta',  # 8
    'postcava',  # 9: inferior vena cava
    'psv',  # 10: portal vein and splenic vein
    'pancreas',  # 11
    'radrenal',  # 12
    'ladrenal'  # 13
)
# 03_CHAOS
CHAOS_TEMPLATE = Template(
    'liver'
)
# 04_LiTS
LiTS_TEMPLATE = Template(
    'liver',
    'liver_tumor'
)
# 05_KiTS
KiTS_TEMPLATE = Template(
    'rkidney',
    'kidney_tumor',
    'kidney_cyst',
    'lkidney'
)
# 08_AbdomenCT-1K
AbdomenCT1K_TEMPLATE = Template(
    'liver',
    'rkidney',
    'spleen',
    'pancreas',
    'lkidney'
)
# 09_AMOS
AMOS_TEMPLATE = Template(
    'spleen',
    'rkidney',
    'lkidney',
    'gall',  # gallbladder
    'esophagus',
    'liver',
    'stomach',
    'aorta',
    'postcava',  # inferior vena cava
    'pancreas',
    'radrenal',
    'ladrenal',
    'duodenum',
    'bladder',
    'prost_ut',  # prostate/uterus
)
# 10_Task03_Liver
DEC_Task03_Liver_TEMPLATE = Template(
    'liver',
    'liver_tumor'
)
# 10_Task06_Lung
DEC_Task06_Lung_TEMPLATE = Template('lung_tumor')
# 10_Task07_Pancreas
DEC_Task07_Pancreas_TEMPLATE = Template(
    'pancreas',
    'pancreas_tumor'
)
# 10_Task08_HepaticVessel
DEC_Task08_HepaticVessel_TEMPLATE = Template(
    'hepatic_vessels',  # Hepatic Vessel
    'hepatic_tumor'
)
# 10_Task09_Spleen
DEC_Task09_Spleen_TEMPLATE = Template('spleen')
# 10_Task10_Colon
DEC_Task10_Colon_TEMPLATE = Template('colon_tumor')
# 12_CT-ORG
CTORG_TEMPLATE = Template(
    'liver',
    'bladder',
    'rlung',
    'rkidney',
    'bone',
    'brain',
    'llung',
    'lkidney'
)
# 13_AbdomenCT-12organ
AbdomenCT12_TEMPLATE = Template(
    'liver',
    'rkidney',
    'spleen',
    'pancreas',
    'aorta',
    'postcava',
    'stomach',
    'gall',
    'esophagus',
    'radrenal',
    'ladrenal',
    'celiac_truck',  # celiac artery
    'lkidney'
)


def get_template_for_name(name):
    # 01. BTCV (Multi-atlas)
    if 'Multi-Atlas' in name:
        return BTCV_TEMPLATE
    # 03. CHAOS
    if 'CHAOS' in name:
        return CHAOS_TEMPLATE
    # 04_LiTS
    if 'LiTS' in name:
        return LiTS_TEMPLATE
    # 05_KiTS
    if 'KiTS' in name:
        return KiTS_TEMPLATE
    # 08_AbdomenCT-1K
    if 'AbdomenCT-1K' in name:
        return AbdomenCT1K_TEMPLATE
    # 09_AMOS
    if 'AMOS' in name:
        return AMOS_TEMPLATE
    # 10_Task03_Liver
    if 'Task03' in name:
        return DEC_Task03_Liver_TEMPLATE
    # 10_Task06_Lung
    if 'Task06' in name:
        return DEC_Task06_Lung_TEMPLATE
    # 10_Task07_Pancreas
    if 'Task07' in name:
        return DEC_Task07_Pancreas_TEMPLATE
    # 10_Task08_HepaticVessel
    if 'Task08' in name:
        return DEC_Task08_HepaticVessel_TEMPLATE
    # 10_Task09_Spleen
    if 'Task09' in name:
        return DEC_Task09_Spleen_TEMPLATE
    # 10_Task10_Colon
    if 'Task10' in name:
        return DEC_Task10_Colon_TEMPLATE
    # 12_CT-ORG
    if 'CT-ORG' in name:
        return CTORG_TEMPLATE
    # 13_AbdomenCT-12organ
    if 'AbdomenCT-12organ' in name:
        return AbdomenCT12_TEMPLATE

    raise ValueError("WARNING: No template found for " + name)
