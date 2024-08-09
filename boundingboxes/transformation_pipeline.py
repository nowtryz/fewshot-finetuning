import torch
from monai.apps.detection.transforms.dictionary import RandCropBoxByPosNegLabeld, RandZoomBoxd, RandRotateBox90d
from monai.transforms import (Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld, SaveImaged, SelectItemsd, DeleteItemsd,
                              EnsureChannelFirstd)

from pretrain.datasets.transforms import (CategoricalToOneHot, MapLabels, LRDivision,
                                          MatchTemplate, GetAnnotationMask, ApplyToDecathlonOnly)
from utils.templates import NUM_CLASSES
from .transforms import (DegradeToBoundingBoxesD, BoundingBoxesToOneHotD)


def make_bb_augmentation_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        MatchTemplate(destination_key="template"),  # Match dataset to according template
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Enforce orientation to correctly separate Left/Right
        LRDivision(template_key="template", image_key="image", label_key="label"),  # Separate left and right organs
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True, dtype=None),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
        SpatialPadd(keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        # Degrade copied label to bounding boxes
        DegradeToBoundingBoxesD(
            keys="label",
            box_keys="boxes",
            box_labels_keys="boxes_labels",
            template_key="template"
        ),
        MapLabels(keys=['label', 'boxes_labels']),  # Map original datasets labels to universal label
        # Perform data augmentation
        ApplyToDecathlonOnly(  # is it treated as random ???
            RandZoomBoxd(
                image_keys=["image", "label"],
                box_keys="boxes",
                box_ref_image_keys="image",
                mode=['area', 'nearest'],
                prob=0.3, min_zoom=1.1, max_zoom=1.3
            )
        ),
        RandCropBoxByPosNegLabeld(image_keys=["image", "label"], box_keys="boxes", label_keys="boxes_labels",
                                  spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                                  num_samples=args.num_samples, image_threshold=0),
        RandRotateBox90d(image_keys=["image", "label"], box_keys="boxes", box_ref_image_keys="image",
                         prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        # Prepare data for training
        MatchTemplate(),  # Match dataset to according template once again to be able to retrieve annotation mask
        GetAnnotationMask(),  # Copy annotation mask from template to a separate key
        ToTensord(keys=["annotation_mask", "boxes_labels", "boxes_labels"], track_meta=False),  # Drop MetaTensor
        ToTensord(keys=["image", "label"], dtype=torch.float32, track_meta=False),
        # Adapt bounding boxes for losses
        BoundingBoxesToOneHotD(boxes_keys='boxes', labels_keys='boxes_labels', box_ref_image_keys='image',
                               num_classes=NUM_CLASSES, sparse=True),
        SelectItemsd(keys=["image", "label", "annotation_mask", "boxes"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])


def make_bb_validation_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        MatchTemplate(destination_key="template"),  # Match dataset to according template
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Enforce orientation to correctly separate Left/Right
        LRDivision(template_key="template", image_key="image", label_key="label"),  # Separate left and right organs
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True, dtype=None),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
        SpatialPadd(keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        MapLabels(keys=['label']),  # Map original datasets labels to universal label
        # Perform a random crop to fit image on GPU Memory
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        # Prepare data for validation
        GetAnnotationMask(),  # Copy annotation mask from template to a separate key
        ToTensord(keys=["annotation_mask"]),  # Drop MetaTensor
        ToTensord(keys=["image", "label"], dtype=torch.float32),
        SelectItemsd(keys=["image", "label", "annotation_mask"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])
