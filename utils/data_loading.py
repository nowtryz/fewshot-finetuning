from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import ignite
import torch.multiprocessing
from ignite.distributed import one_rank_only
from monai.data import Dataset, DistributedSampler, DataLoader, list_data_collate
from monai.transforms import apply_transform
from torch.utils.data import random_split
from tqdm import tqdm


def get_train_data(args, root: Path):
    # training dict part
    data_dicts_train = []
    for line in open(args.data_txt_path):
        image, label = line.strip().split()
        name = label.split('.')[0]
        data_dicts_train.append({
            'image': root / image,
            'label': root / label,
            'name': name,
            'bounding_box': root / f'{name}_bounding-boxes.nii.gz'  # Just for bounding box experiments
        })

    return data_dicts_train


@one_rank_only(with_barrier=True)
def preprocess_data(args, base_transform):
    data = get_train_data(args, args.data_root_path)

    print(f'train len {len(data)}')

    if 0 <= args.max_preprocessor <= 1:
        for item in tqdm(data, desc="preprocessing train data"):
            apply_transform(base_transform, item)
    else:
        max_workers = args.max_preprocessor if args.max_preprocessor != -1 else None
        workers = min(max_workers or cpu_count(), cpu_count(), len(data))

        print(f'Preprocessing data with {workers} workers')

        with ProcessPoolExecutor(max_workers=workers, mp_context=torch.multiprocessing.get_context()) as pool:
            for _ in tqdm(pool.map(base_transform, data), desc="preprocessing train data", total=len(data)):
                pass


def _make_loader(args, data, transforms, collate_fn, batch_size):
    dataset = Dataset(data=data, transform=transforms)
    sampler = DistributedSampler(dataset=dataset, even_divisible=True,
                                 shuffle=args.shuffle) if args.dist else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle and sampler is None,
                        num_workers=args.num_workers, collate_fn=collate_fn or list_data_collate,
                        sampler=sampler, pin_memory=args.pin_memory)

    return loader


def get_loader(args, transforms, val_transforms, collate_fn=None):
    data_dicts_train = get_train_data(args, args.preprocessed_output)
    train_data, val_data = random_split(
        Dataset(data_dicts_train),
        [args.train_val_split, 1 - args.train_val_split]
    )

    train_loader = _make_loader(args, train_data, transforms, collate_fn, args.batch_size)
    val_loader = _make_loader(args, val_data, val_transforms, collate_fn, args.val_batch_size)

    return train_loader, val_loader
