import argparse
import os
from pathlib import Path


class FloatRangeAction(argparse.Action):
    def __init__(self, option_strings, dest, type=float, minimum=None, maximum=None, nargs=None, **kwargs):  # noqa
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, type=type, nargs=1, **kwargs)
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, parser, namespace, values, option_string=None):
        value = values[0]
        if self.minimum is not None and value < self.minimum:
            raise argparse.ArgumentTypeError(f"Must be a floating point number >= {self.minimum}, got {value}")
        if self.maximum is not None and value > self.maximum:
            raise argparse.ArgumentTypeError(f"Must be a floating point number ><= {self.maximum}, got {value}")

        setattr(namespace, self.dest, value)


def make_main_parser():
    parser = argparse.ArgumentParser()

    # Folders, dataset, etc.
    parser.add_argument('--out-path', default='./pretrain/results/', type=Path, help='The path resume from checkpoint')
    parser.add_argument('--data-root-path', default="./data/", type=Path, help='data root path')
    parser.add_argument('--stage', default="train", help='train/val')
    parser.add_argument('--data-txt-path', default='./pretrain/datasets/train.txt', help='data txt path')
    parser.add_argument('--preprocessed-output', default='./data/preprocessed-data', type=Path,
                        help='Locations to store cached preprocessed images and labels')
    parser.add_argument('--use-cache', action='store_true', help='Use previously preprocessed images')
    parser.add_argument('--preprocess-only', action='store_true', help="Don't train, and only perform the "
                                                                       "preprocessing")
    parser.add_argument('--max-preprocessor', type=int, default=-1, help='Number of cores to use for preprocessing,'
                                                                         ' -1 to use as many as available')

    # Training options
    parser.add_argument('--max-epoch', default=800, type=int, help='Number of training epochs')
    parser.add_argument('--store-num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--classifier', default='linear', help='type of classifier')
    parser.add_argument('--text-controller-type', default='word', help='type of text controller')
    parser.add_argument('--batch-size', default=2, type=int, help='batch size')
    parser.add_argument('--val-batch-size', default=4, type=int, help='batch size during validation')
    parser.add_argument('--train-val-split', default=.9, action=FloatRangeAction, minimum=.1, maximum=1.,
                        help='Ratio of training data with respect to validation data')
    parser.add_argument('--val-epochs', default=1, type=int, help='number of epoch between validation steps')
    parser.add_argument('--unbalanced', action='store_false', dest='balanced')  # balanced defaults to True
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle')  # shuffle defaults to True
    parser.add_argument('--num-samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--pretrained_model',
                        default='./pretrain/pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt',
                        help='The path of pretrain model')

    # Resources
    parser.add_argument('--no-pin-memory', action='store_false', dest='pin_memory',
                        help='Avoid copying Tensors into GPU pinned memory before transferring them')
    parser.add_argument('--dist', action='store_true', help='distributed training or not')
    parser.add_argument("--device")
    parser.add_argument('--num-workers', default=1, type=int, help='workers number for DataLoader')
    parser.add_argument('--disable-comet', action='store_true', help='Disable the comet logger for this run')

    # Volume pre-processing
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='Size of the cropped ROI in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='Size of the cropped ROI in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='Size of the cropped ROI in z direction')

    # Resume training options
    parser.add_argument('--resume', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--resume_model_id', default=None)
    parser.add_argument('--last_epoch', default=1, type=int)

    return parser
