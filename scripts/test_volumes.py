import argparse
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from sys import stderr

import numpy as np
import torch
from monai.transforms import LoadImageD
from scipy import ndimage
from torch.nn import functional as F  # noqa
from tqdm import tqdm

from boundingboxes.sparse_utils import sparse_pad_sequence
from utils.data_loading import get_train_data
from utils.decorators import wrap_ndarray_in_tensor
from utils.misc import print_monitoring


@wrap_ndarray_in_tensor
def _3d_segmentation_to_bounding_boxes(segmentation_mask: np.ndarray) -> np.ndarray:
    """W x H x D -> N x W x H x D"""
    if not np.any(segmentation_mask):
        return np.zeros((0, *segmentation_mask.shape), dtype=np.bool_)

    # Run scipy nd-image's connected components algorithm
    labels, num_features = ndimage.label(segmentation_mask)
    objects = ndimage.find_objects(labels)
    result = np.zeros((num_features, *segmentation_mask.shape), dtype=np.bool_)

    assert len(objects) == num_features  # noqa

    for component, slices in enumerate(objects):
        # Selecting component's slice and computed slices
        result[(component,) + slices] = True

    return result


def degrade_to_bounding_boxes(volume: torch.Tensor):
    hot = F.one_hot(volume.long()).permute(-1, 0, 1, 2).bool()
    classes_present = torch.flatten(hot, start_dim=-3).any(dim=-1)
    hot_few = hot[classes_present]
    # del classes_present
    # del hot

    classes = [
        _3d_segmentation_to_bounding_boxes(class_).to_sparse(1)
        for class_ in torch.unbind(hot_few, dim=0)
    ]

    return sparse_pad_sequence(classes).bool()


def get_info(folder: Path, label, name, **_):
    # Compute primitives
    num_classes = len(torch.unique(label)) - 1

    # Check cache before running
    dest = folder / (name + '-bb.pt')
    if dest.exists():
        bounding_boxes = torch.load(dest)
    else:
        bounding_boxes = degrade_to_bounding_boxes(label)  # C* x N* x H x W x D (* = sparse)
        # Save produced bounding boxes
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bounding_boxes, dest)

    # Perform in depth analysis
    overlapping_bb: bool = torch.any(
        torch.sparse.sum(bounding_boxes, dim=1)
        .values()
        > 1
    ).item()
    class_masks = torch.sparse.sum(bounding_boxes, dim=1).bool()  # Any::dim not supported on sparse backend
    overlapping_classes: bool = torch.any(torch.sparse.sum(class_masks, dim=0) > 1).item()  # noqa

    return num_classes, overlapping_bb, overlapping_classes


def worker_function(loader, folder: Path, initial_data):
    try:
        data = loader(initial_data)
        return get_info(folder, **data)
    except Exception: # noqa
        print(f"An error occurred when processing {initial_data}", file=stderr)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-txt-path', default='./pretrain/datasets/partial.txt', type=Path, help='data txt path')
    parser.add_argument('--data-root-path', default="./data/", type=Path, help='data root path')
    parser.add_argument('--num-workers', default=1, type=int, help="Number of parallel processes")
    parser.add_argument('--cache-location', '--output-path', '-O', default='./data/bb', type=Path,
                        help="Destination of the cached degraded labels")
    args = parser.parse_args()

    data_dicts_train = get_train_data(args, args.data_root_path)
    image_loader = LoadImageD(keys=["image", "label"])
    worker = partial(worker_function, image_loader, args.cache_location)

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        print(f"Processing with {args.num_workers} workers")
        mapping = pool.map(worker, data_dicts_train)
        progress = tqdm(mapping, desc="Parsing images", total=len(data_dicts_train))
        accumulation = list(progress)
        accumulation = filter(lambda x: x is not None, accumulation)

    num_classes_list, overlapping_bb_list, overlapping_classes_list = zip(*accumulation)

    print("num_classes:", Counter(num_classes_list))
    print("overlapping_bb:", Counter(overlapping_bb_list))
    print("overlapping_classes:", Counter(overlapping_classes_list))
    print_monitoring()
