import argparse
import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Dict, cast, Any, Concatenate, ParamSpec, TypeVar, Union, Literal

import comet_ml.integration.pytorch
import torch
from ignite import distributed as idist
from ignite.contrib.engines.common import setup_common_training_handlers
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import ProgressBar, global_step_from_engine
from ignite.metrics import Metric
from torch import distributed as dist, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from boundingboxes.losses import LossWrapper
from utils.comet_logger import CometLogger
from utils.data_loading import preprocess_data, get_loader
from utils.misc import set_seeds
from utils.models import load_weights, default_model
from utils.scheduler import LinearWarmupCosineAnnealingLR
from utils.templates import NUM_CLASSES


@dataclass
class Experiment:
    args: argparse.Namespace
    out_path: Path
    model: nn.Module
    num_classes: int
    rank: int
    local_rank: int
    device: torch.device
    optimizer: Optimizer
    scheduler: LRScheduler
    train_loader: DataLoader
    val_loader: DataLoader
    comet: Optional[CometLogger]

    def is_main_process(self) -> bool:
        """
        Return True if the current process is the main process in a distributed training.
        Always True is not distributed
        """
        return self.rank == 0

    def __enter__(self):
        global _CURRENT_EXPERIMENT
        _CURRENT_EXPERIMENT = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_EXPERIMENT
        _CURRENT_EXPERIMENT = None

        # Stop distributed training if applicable
        idist.finalize()

        # Close the comet logger if present on this node
        if self.comet is not None:
            self.comet.close()
        return False


# Decorators to use as guards
_CURRENT_EXPERIMENT: Optional[Experiment] = None
P = ParamSpec('P')
R = TypeVar('R')


def use_current_experiment(func: Callable[Concatenate[Experiment, P], R]) -> Callable[P, R]:
    """Decorate a function requiring the experiment from the current context"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _CURRENT_EXPERIMENT
        if _CURRENT_EXPERIMENT is None:
            raise LookupError("No experiment set up")
        return func(_CURRENT_EXPERIMENT, *args, **kwargs)

    return wrapper


def comet_logging(func):
    """
    Decorate a function requiring the comet logger to log information and is not relevant if the logger is not present
    """

    @functools.wraps(func)
    @use_current_experiment
    def wrapper(experiment: Experiment, *args, **kwargs):
        if experiment.comet is None:
            return
        return func(experiment.comet, *args, **kwargs)

    return wrapper


# Factories

@idist.one_rank_only()
def _setup_comet_logger(model, args):
    """Create comet logger"""
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name=os.environ.get("COMET_PROJECT_NAME"),
        auto_metric_logging=False,
        auto_output_logging="simple",
        parse_args=False,
        disabled=args.disable_comet
    )

    comet_logger.set_model_graph(model)
    comet_logger.log_parameters(vars(args))  # logs default and overridden arguments
    comet_ml.integration.pytorch.watch(model)

    if 'COMET_EXPERIMENT_NAME' in os.environ:
        comet_logger.experiment.set_name(os.environ['COMET_EXPERIMENT_NAME'])

    if dist.is_initialized():
        comet_logger.log_parameters({
            'dist/device_count': dist.get_world_size(),
            'dist/true_batch_size': dist.get_world_size() * args.batch_size,
        })

    return comet_logger


def setup_experiment(
        # Give all arguments as keyword arguments to ensure edits don't break compatibility with existing experiments
        *,
        args: argparse.Namespace,
        preprocessing_transforms: Callable,
        augmentation_transforms: Callable,
        validation_transforms: Callable,
        num_classes: int = NUM_CLASSES,
        model: nn.Module = None,
        collate_fn=None
):
    if not args.use_cache:
        preprocess_data(args, preprocessing_transforms)
    if args.preprocess_only:
        sys.exit(0)

    if args.dist:
        idist.initialize(backend="nccl")
        idist.show_config()

    # Set seeds for reproducibility
    rank = idist.get_rank()
    set_seeds(42 + rank)

    if model is None:
        model = default_model(args, num_classes)

    out_path: Path = args.out_path

    # If distributed, cuda device has been set by idist.initialize(...)
    device = idist.device()
    print("Using device", device)

    if rank == 0 and not out_path.exists():
        out_path.mkdir(parents=True)

    # Load pretrained weights from encoder and set device
    if args.resume:
        model = load_weights(model, args.resume_model_id, classifier=True)
    else:
        model = load_weights(model, args.pretrained_model)

    model = idist.auto_model(model)

    # Set optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch,
                                              max_epochs=args.max_epoch,
                                              warmup_start_lr=args.lr / args.warmup_epoch)

    train_loader, val_loader = get_loader(args, augmentation_transforms,
                                          validation_transforms, collate_fn)

    args.NUM_CLASS = num_classes

    return Experiment(
        args, out_path,
        model, num_classes,
        rank, idist.get_local_rank(), device,
        optimizer, scheduler,
        train_loader,
        val_loader,
        _setup_comet_logger(model, args),
    )


@use_current_experiment
def setup_trainer(experiment: Experiment, trainer: Engine, metric_names="all", **kwargs):
    args = experiment.args
    with_pbars = kwargs.pop('with_pbars', True)

    setup_common_training_handlers(
        trainer=trainer,
        train_sampler=cast(DistributedSampler, experiment.train_loader.sampler),
        output_path=str(experiment.out_path),
        lr_scheduler=experiment.scheduler,
        with_pbars=False,
        log_every_iters=1,
        **kwargs
    )

    if with_pbars:
        ProgressBar(persist=True).attach(trainer, metric_names=metric_names)

    @trainer.on(Events.EPOCH_STARTED)
    def update_args_epoch(engine: Engine) -> None:
        args.epoch = engine.state.epoch

    @trainer.on(Events.EPOCH_COMPLETED(every=args.store_num))
    @idist.one_rank_only(with_barrier=True)  # Make all rank wait for the main process to save before continuing
    def store_model():
        torch.save(experiment.model.state_dict(), experiment.out_path / f'pretrained_epoch{args.epoch}.pth')
        print('saved model successfully')


@use_current_experiment
def setup_evaluator(
        experiment: Experiment,
        trainer: Engine,
        evaluator: Engine = None,
        metrics: Dict[str, Metric] = None,
        *args, **kwargs,
):
    if evaluator is None:
        evaluator = create_supervised_evaluator(
            model=experiment.model,
            metrics=metrics,
            device=experiment.device,
            *args, **kwargs,
        )
    elif metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    ProgressBar(persist=True, desc='Evaluation').attach(evaluator, metric_names='all')

    @trainer.on(Events.EPOCH_COMPLETED(every=experiment.args.val_epochs))
    def log_validation_results():
        evaluator.run(experiment.val_loader)

    return evaluator


def global_epoch_from_engine(engine: Engine):
    def transform(_: Any):
        return engine.state.epoch
    return transform


@use_current_experiment
@idist.one_rank_only()
def setup_logger(
        experiment: Experiment,
        trainer: Engine,
        evaluator: Engine = None,
        output_transform: Callable[[LossWrapper], dict] = None,
        training_metrics: Optional[Union[List[str], Literal['all']]] = "all",
        epoch_training_metrics: Optional[List[str]] = None,
        validation_metrics: Optional[List[str]] = "all",
):
    if experiment.comet is None:
        return

    experiment.comet.log_parameters({
        'train-size': len(experiment.train_loader),
        'val-size': len(experiment.val_loader) if experiment.val_loader is not None else 0,
    })

    # Attach logger to trainer
    experiment.comet.attach_output_handler(
        trainer, Events.ITERATION_COMPLETED,
        tag="training",
        metric_names=training_metrics,
        output_transform=output_transform
    )
    experiment.comet.attach_opt_params_handler(trainer, Events.ITERATION_STARTED, optimizer=experiment.optimizer)

    # Attach logger to evaluator if present
    if evaluator is not None:
        # Validation, > 0 <=> epoch % 2 == 1
        experiment.comet.attach_output_handler(
            evaluator, Events.COMPLETED, tag="validation",
            metric_names=validation_metrics,
            global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
            global_epoch_transform=global_epoch_from_engine(trainer)
        )

    if epoch_training_metrics is not None and training_metrics != 'all':
        experiment.comet.attach_output_handler(
            trainer, Events.EPOCH_COMPLETED, tag='training',
            metric_names=epoch_training_metrics,
            global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
            global_epoch_transform=global_epoch_from_engine(trainer),
        )
