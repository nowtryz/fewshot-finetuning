from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from monai.data import DataLoader, Dataset, DistributedSampler, list_data_collate
from monai.transforms import (AddChanneld, Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld, SaveImaged, apply_transform, SelectItemsd)
from tqdm import tqdm

from pretrain.datasets.utils import CategoricalToOneHot, SelectRelevantKeys, MapLabels, \
    LRDivision, UniformDataset, RandZoomd_select


def get_train_data(args, root: Path):
    # training dict part
    data_dicts_train = []
    for iPartition in args.partitions:
        for line in open(args.data_txt_path[iPartition]):
            image, label = line.strip().split()
            data_dicts_train.append({
                'image': root / image,
                'label': root / label,
                'name': label.split('.')[0],
            })

    return data_dicts_train


def preprocess_data(args):
    data = get_train_data(args, args.data_root_path)
    base_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        LRDivision(),
        SaveImaged(keys=["image", "label"], output_dir=args.preprocessed_output, output_postfix='', resample=False,
                   data_root_dir=args.data_root_path, separate_folder=False, print_log=False),
        SelectItemsd(keys=['image', 'label', 'name']),
    ])

    print(f'train len {len(data)}')

    if args.num_workers <= 1:
        for item in tqdm(data, desc="preprocessing train data"):
            apply_transform(base_transform, item)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for _ in tqdm(pool.map(base_transform, data), desc="preprocessing train data", total=len(data)):
                pass


def get_loader(args):

    # Prepare data transforms and augmentations
    transforms = Compose([
        # Load preprocessed images
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        # Perform data augmentation
        RandZoomd_select(keys=["image", "label"], prob=0.3, min_zoom=1.1, max_zoom=1.3, mode=['area', 'nearest']),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        # Prepare data for training
        SelectItemsd(keys=['image', 'label', 'name']),
        MapLabels(),
        ToTensord(keys=["image", "label"]),
        CategoricalToOneHot(args.NUM_CLASS+1),
    ])

    data_dicts_train = get_train_data(args, args.preprocessed_output)
    # data_dicts_train = get_train_data(args, args.data_root_path)

    if args.balanced:
        train_dataset = UniformDataset(data=data_dicts_train, transform=transforms)
    else:
        train_dataset = Dataset(data=data_dicts_train, transform=transforms)

    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=args.shuffle) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle and train_sampler is None,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              sampler=train_sampler, pin_memory=args.pin_memory)

    return train_loader, train_sampler
