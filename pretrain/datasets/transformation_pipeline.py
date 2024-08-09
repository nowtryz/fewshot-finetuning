from monai.transforms import (Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld, SaveImaged, SelectItemsd, EnsureChannelFirstd, RandZoomd)

from pretrain.datasets.transforms import (CategoricalToOneHot, MapLabels, LRDivision, MatchTemplate,
                                          GetAnnotationMask, ApplyToDecathlonOnly)
from utils.templates import NUM_CLASSES


def make_augmentation_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        MatchTemplate(),
        LRDivision(),
        SaveImaged(keys=["image", "label"], output_dir=args.preprocessed_output, output_postfix='', resample=False,
                   data_root_dir=args.data_root_path, separate_folder=False, print_log=False),
        SelectItemsd(keys=['image', 'label', 'name']),
        # Perform data augmentation
        ApplyToDecathlonOnly(
            RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=1.1, max_zoom=1.3, mode=['area', 'nearest'])
        ),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        # Prepare data for training
        SelectItemsd(keys=['image', 'label', 'name']),
        MatchTemplate(),
        MapLabels(),
        GetAnnotationMask(),
        ToTensord(keys=["image", "label"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])


def make_validation_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        MatchTemplate(),
        LRDivision(),
        # Perform a random crop to fit image on GPU Memory
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        # Prepare data for validation
        SelectItemsd(keys=['image', 'label', 'name']),
        MapLabels(),
        GetAnnotationMask(),
        ToTensord(keys=["image", "label"]),
        CategoricalToOneHot(NUM_CLASSES),
    ])


