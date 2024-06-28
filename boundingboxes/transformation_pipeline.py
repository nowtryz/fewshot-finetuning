import torch
from monai.transforms import (Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld, SaveImaged, SelectItemsd, CopyItemsd, DeleteItemsd,
                              EnsureChannelFirstd)

from pretrain.datasets.transforms import (CategoricalToOneHot, MapLabels, LRDivision,
                                          RandZoomd_select, MatchTemplate, GetAnnotationMask)
from utils.templates import NUM_CLASSES
from .transforms import (OriginalDimToUniversalDim, CategoricalToOneHotDynamic,
                         AsDictionaryTransform, DegradeToBoundingBoxes, BoundingBoxesToOneHot, DegradeToBoundingBoxesD)


def make_bb_preprocessing_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        MatchTemplate(destination_key="template"),  # Match dataset to according template
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Enforce orientation to correctly separate Left/Right
        LRDivision(template_key="template", image_key="image", label_key="label"),  # Separate left and right organs
        DeleteItemsd("bounding_box"),  # "bounding_box" is already present, containing its storing location
        CopyItemsd(keys=["label"],
                   names=["bounding_box"]),  # Copy label to a new key
        CategoricalToOneHotDynamic(template_key="template", label_key="bounding_box"),  # Convert copied key to one hot
        # Degrade copied label to bounding boxes
        DegradeToBoundingBoxesD(keys=["bounding_box"], template_key="template"),
        MapLabels(),  # Map original datasets labels to universal label (using 'template', affects 'label')
        Spacingd(keys=["image", "label", "bounding_box"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label", "bounding_box"], source_key="image", allow_smaller=False),
        SpatialPadd(keys=["image", "label", "bounding_box"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        SaveImaged(keys=["image", "label"],
                   output_dir=args.preprocessed_output, output_postfix='', resample=False,
                   data_root_dir=args.data_root_path, separate_folder=False, print_log=False),
        SaveImaged(keys=["bounding_box"],
                   output_dir=args.preprocessed_output, output_postfix='bounding-boxes', resample=False,
                   data_root_dir=args.data_root_path, separate_folder=False, print_log=False),
        SelectItemsd(keys=["image", "label", "name", "bounding_box"]),
    ])


def make_bb_augmentation_transforms(args):
    return Compose([
        # Load preprocessed images
        LoadImaged(keys=["image", "label", "bounding_box"]),
        EnsureChannelFirstd(keys=["image", "label", "bounding_box"]),
        # Perform data augmentation
        RandZoomd_select(keys=["image", "label", "bounding_box"],
                         mode=['area', 'nearest', 'nearest'],
                         prob=0.3, min_zoom=1.1, max_zoom=1.3),
        RandCropByPosNegLabeld(keys=["image", "label", "bounding_box"], label_key="bounding_box",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        RandRotate90d(keys=["image", "label", "bounding_box"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        # Prepare data for training
        MatchTemplate(),  # Match dataset to according template once again to be able to retrieve annotation mask
        GetAnnotationMask(),  # Copy annotation mask from template to a separate key
        OriginalDimToUniversalDim(template_key="template",
                                  bb_key="bounding_box"),  # Adapt bounding boxes to universal dimensions
        ToTensord(keys=["annotation_mask", "bounding_box"], track_meta=False),  # Drop MetaTensor
        ToTensord(keys=["image", "label"], dtype=torch.float32, track_meta=False),
        AsDictionaryTransform(BoundingBoxesToOneHot(sparse=True),  # Adapt bounding boxes for losses
                              keys=["bounding_box"]),
        SelectItemsd(keys=["image", "label", "annotation_mask", "bounding_box"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])


def make_bb_validation_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Perform a random crop to fit image on GPU Memory
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        # Prepare data for validation
        MatchTemplate(),  # Match dataset to according template once again to be able to retrieve annotation mask
        GetAnnotationMask(),  # Copy annotation mask from template to a separate key
        ToTensord(keys=["annotation_mask"]),  # Drop MetaTensor
        ToTensord(keys=["image", "label"], dtype=torch.float32),
        SelectItemsd(keys=["image", "label", "annotation_mask"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])
