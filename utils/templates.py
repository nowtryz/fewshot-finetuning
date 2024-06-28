# Unified template
from collections import UserDict
from typing import Dict

import numpy as np
import torch

UNIVERSAL_TEMPLATE = {'spleen': 1, 'rkidney': 2, 'lkidney': 3, 'gall': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7,
                      'aorta': 8, 'postcava': 9, 'psv': 10, 'pancreas': 11, 'radrenal': 12, 'ladrenal': 13,
                      'duodenum': 14, 'bladder': 15, 'prost_ut': 16, 'liver_tumor': 17, 'kidney_tumor': 18,
                      'kidney_cyst': 19, 'celiac_truck': 20, 'rlung': 21, 'llung': 22, 'bone': 23, 'brain': 24,
                      'lung_tumor': 25, 'pancreas_tumor': 26, 'hv': 27, 'hvt': 28, 'colon_tumor': 29}

NUM_CLASSES = np.max(np.fromiter(UNIVERSAL_TEMPLATE.values(), dtype=int)) + 1


class Template(UserDict):
    def __init__(self, *class_labels: str):
        super().__init__({label: idx + 1 for idx, label in enumerate(class_labels)})
        self.num_classes = len(class_labels) + 1

        # Build indices: allow converting original indices to universal indices with advanced indexing
        # -1 in the reversed index is used to yield an empty class, as the background will be filled with some values
        self._mapping_index = np.zeros(self.num_classes)  # use to convert categorical labels
        self._reverse_index = -np.ones(NUM_CLASSES)  # use to convert one-matrices
        self._reverse_index[0] = 0  # Match backgrounds
        for category, original_idx in self.items():
            universal_idx = UNIVERSAL_TEMPLATE[category]
            self._reverse_index[universal_idx] = original_idx
            self._mapping_index[original_idx] = universal_idx

        self.annotation_mask = np.array(self._reverse_index != -1).astype(int)
        self._mapping_index_torch = torch.tensor(self._mapping_index)

    def __call__(self, volume):
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
        pad = torch.zeros(1, *volume.shape[1:])  # Create a blank class with the volume dimension
        volume = torch.cat([volume, pad])  # Append pad as last element, so that reversed index `-1` yields the padding
        return volume[self._reverse_index]


# 01. BTCV (Multi-atlas)
BTCV_TEMPLATE = Template(
    'spleen',
    'rkidney',
    'lkidney',
    'gall',
    'esophagus',
    'liver',
    'stomach',
    'aorta',
    'postcava',
    'psv',
    'pancreas',
    'radrenal',
    'ladrenal'
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
    'gall',
    'esophagus',
    'liver',
    'stomach',
    'aorta',
    'postcava',
    'pancreas',
    'radrenal',
    'ladrenal',
    'duodenum',
    'bladder',
    'prost_ut'
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
    'hv',
    'hvt'
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
    'celiac_truck',
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

    raise ValueError("WARNING: Not template found for " + name)
