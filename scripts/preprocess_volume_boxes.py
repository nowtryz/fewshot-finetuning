import argparse
import logging
import tarfile
from pathlib import Path

from monai.transforms import apply_transform, RandomizableTrait

from boundingboxes import make_bb_augmentation_transforms
from utils.data_loading import get_train_data
from utils.parser import register_preprocessing_arguments, register_folder_arguments


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)-7s] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('volume', type=int, help='Index of the volume from the computed train data')
    parser.add_argument('--archive', type=argparse.FileType('rb'),
                        help='path to the dataset archive, if the script need to extract file')
    register_folder_arguments(parser)
    register_preprocessing_arguments(parser)
    args = parser.parse_args()

    preprocessing_transforms = make_bb_augmentation_transforms(args)
    first_random = preprocessing_transforms.get_index_of_first(lambda t: isinstance(t, RandomizableTrait))
    item = get_train_data(args.data_txt_path, args.data_root_path)[args.volume]

    if args.archive is not None:
        root = args.data_root_path

        with args.archive, tarfile.open(fileobj=args.archive) as archive:
            logging.info('Looking into %s...', args.archive.name)
            archive_image = archive.getmember(('data' / item['image'].relative_to(root)).as_posix())
            archive_label = archive.getmember(('data' / item['label'].relative_to(root)).as_posix())

            extract(archive, archive_image, item['image'])
            extract(archive, archive_label, item['label'])

    logging.info(f"Preprocessing {item['bounding_box']}...")
    preprocessing_transforms(item, end=first_random)


def extract(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path):
    # Files will be fully loaded in memory during processing, hence the provided resources are supposed to be
    # enough to read entire files to memory for archiving, no need for buffered read/write
    logging.info('Inflating "%s" to "%s"...', member.name, destination)
    with archive.extractfile(member) as image:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image.read())


if __name__ == '__main__':
    main()
