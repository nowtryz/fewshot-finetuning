import numpy as np
import torch

from monai.transforms import (RandZoomd, Transform, ThreadUnsafe)

from utils.templates import get_template_for_name, Template


class LRDivision(Transform, ThreadUnsafe):
    def __init__(self, template_key='template', image_key='image', label_key='label'):
        self.template_key = template_key
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        template: Template = data[self.template_key]

        if 'rkidney' in template or 'rlung' in template:
            center_orig = np.array(data[self.image_key].shape) // 2

            if 'rkidney' in template:

                # rkidney --> lkidney
                mask = data[self.label_key][:, :center_orig[1], :, :] == template['rkidney']
                data[self.label_key][:, :center_orig[1], :, :][mask] = template['lkidney']

            if 'rlung' in template:
                center_orig = np.array(data["image"].shape) // 2

                # rlung --> llung
                mask = data[self.label_key][:, :center_orig[1], :, :] == template['rlung']
                data[self.label_key][:, :center_orig[1], :, :][mask] = template['llung']

        return data


class RandZoomd_select(RandZoomd):
    """Random zoom only for Decathlon"""
    def __call__(self, data, *args, **kwargs):
        d = dict(data)
        name = d['name']
        key = name.split("/")[0]
        if not ("10" in key):
            return d
        d = super().__call__(d, *args, **kwargs)
        return d


class MatchTemplate:
    """
    TODO
    """
    def __init__(self, destination_key='template'):
        self.destination_key = destination_key

    def __call__(self, data):
        data[self.destination_key] = get_template_for_name(data['name'])
        return data


class MapLabels:
    """
    TODO
    """
    def __call__(self, data):
        y = data['label']
        template: Template = data['template']

        try:
            y = template(y)
        except Exception:  # noqa
            raise ValueError("WARNING: Error during mapping. Check that all organs are at the universal template.")

        data['label'] = y
        return data


class GetAnnotationMask:
    """
    TODO
    """
    def __call__(self, data):
        template: Template = data['template']
        data['annotation_mask'] = template.annotation_mask
        return data


class CategoricalToOneHot:
    def __init__(self, classes, label_key='label'):
        self.label_key = label_key
        self.classes = classes

    def __call__(self, data):
        y = data[self.label_key]

        # one hot encoding
        y = torch.nn.functional.one_hot(
            y.to(torch.long).squeeze(0),  # Remove dim created by AddChannel
            num_classes=self.classes
        ).permute((-1, 0, 1, 2))

        data['label'] = y
        return data

