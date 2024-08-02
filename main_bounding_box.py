import logging
import warnings
from operator import itemgetter

import comet_ml  # noqa
import torch
from ignite.engine import Events, Engine
from ignite.handlers import TerminateOnNan
from ignite.metrics import Loss, RunningAverage, Average
from ignite.metrics.metric import RunningBatchWise

from boundingboxes.losses import CombinedLoss, LossWrapper, Reduction
from boundingboxes.sparse_utils import sparse_list_data_collate
from boundingboxes.transformation_pipeline import make_bb_preprocessing_transforms, make_bb_augmentation_transforms, \
    make_bb_validation_transforms
from utils.engine import setup_experiment, setup_trainer, setup_evaluator, setup_logger, Experiment, \
    use_current_experiment
from utils.losses import BinaryDice3D
from utils.misc import capture_duration, monitor_resources, keys_extractor
from utils.parser import make_main_parser


def step_loss(*, loss_fn: CombinedLoss):
    loss_fn.step()


def create_trainer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: CombinedLoss,
        device: torch.device,
        non_blocking: bool = False,
        gradient_accumulation_steps: int = 1,
):
    def update(engine: Engine, batch):
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        model.train()
        x = batch["image"].to(device, non_blocking=non_blocking)
        y = batch["bounding_box"].to(device, non_blocking=non_blocking)
        label = batch["label"].to(device, non_blocking=non_blocking)
        mask = batch["annotation_mask"].to(device, non_blocking=non_blocking)

        with torch.autograd.detect_anomaly():
            y_pred = model(x)
            losses: LossWrapper = loss_fn(y_pred, y, mask)
            loss = losses.weighted_loss

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            if any(torch.isnan(p.data).any() for p in model.parameters()):
                logging.error('Detected NaN in model!')

        return {
            'y_pred': y_pred,
            'y_true': label,
            'boxes': y,
            'mask': mask,
            'tightness_prior_loss': losses.tightness_prior,
            'box_size_loss': losses.box_size,
            'emptiness_constraint_loss': losses.emptiness_constraint,
            'weighted_loss': losses.weighted_loss,
        }

    return Engine(update)


def create_evaluator(model: torch.nn.Module, device: torch.device, non_blocking: bool = False):
    def update(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            x = batch["image"].to(device, non_blocking=non_blocking)
            y = batch["label"].to(device, non_blocking=non_blocking)
            mask = batch["annotation_mask"].to(device, non_blocking=non_blocking)
            y_pred = model(x)

        # move annotation mask to `criterion_kwargs` for the `Loss` metric.
        return y_pred, y, {"annotation_mask": mask}

    return Engine(update)


# Classes selection is not possible as all image may not be from the same dataset and de facto may not have the
# same classes defined.
# Possible solutions are:
#  - if working with each dataset separately, classes can be selected depending on the current dataset
#  - pass relevant classes (annotation_mask) to the losses and use it as a sparse mask against the logits.
#     - ensure our operation are performed on the current class even if no bounding box is present for that
#       class at that moment
#     - other?
#  FIXME Or use masked tensor ? Masked with specified classes so computation only reflects the desired classes
# masked_logits, masked_masks = select_classes(logit_map, y, annotation_mask)


@use_current_experiment
def setup_bb_metrics(experiment: Experiment, trainer: Engine, dice: BinaryDice3D):
    # Attach losses logging and metrics
    Loss(
        dice,
        output_transform=lambda output: (output['y_pred'], output['y_true'], {"annotation_mask": output['mask']}),
        device=experiment.device,
    ).attach(trainer, 'metric/dice')

    losses = [
        ('tightness prior', 'tightness_prior_loss'),
        ('box size', 'box_size_loss'),
        ('emptiness constraint', 'emptiness_constraint_loss'),
        ('weighted sum', 'weighted_loss'),
    ]

    for tag, output_key in losses:
        RunningAverage(
            output_transform=itemgetter(output_key),
            device=experiment.device  # Required to allow reduction over all nodes
        ).attach(trainer, f'loss/running avg {tag}', usage=RunningBatchWise())
        Average(
            output_transform=itemgetter(output_key),
            device=experiment.device  # Required to allow reduction over all nodes
        ).attach(trainer, f'loss/{tag}')


@capture_duration()
@monitor_resources()
def main():
    parser = make_main_parser()
    parser.add_argument('--loss-reduction', default='original', type=Reduction)
    args = parser.parse_args()

    # Memory pinning is not supported by sparse tensor, forcing it to be disabled
    if args.pin_memory:
        warnings.warn("Memory pinning is not supported for bounding boxes as not supported by sparse tensors")
        args.pin_memory = False

    with setup_experiment(
            args=args,
            preprocessing_transforms=make_bb_preprocessing_transforms(args),
            augmentation_transforms=make_bb_augmentation_transforms(args),
            validation_transforms=make_bb_validation_transforms(args),
            collate_fn=sparse_list_data_collate,
    ) as experiment:
        reduction: Reduction = args.loss_reduction
        loss_fn = CombinedLoss(reduction).to(experiment.device)
        dice = BinaryDice3D().to(experiment.device)

        evaluator = create_evaluator(experiment.model, experiment.device)
        trainer = create_trainer(
            model=experiment.model,
            optimizer=experiment.optimizer,
            loss_fn=loss_fn,
            device=experiment.device,
            # TODO would require to customise learning rate scheduler to use accumulation
            # gradient_accumulation_steps=4,
        )

        evaluator.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan(itemgetter(0)))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan(keys_extractor(
            'y_pred', 'tightness_prior_loss', 'box_size_loss', 'emptiness_constraint_loss', 'weighted_loss'
        )))

        setup_trainer(trainer, metric_names=['loss/running avg weighted sum'], stop_on_nan=False)
        setup_evaluator(trainer, evaluator, metrics={
            "metric/dice": Loss(dice)
        })

        setup_bb_metrics(trainer, dice)
        setup_logger(
            trainer, evaluator,
            training_metrics=[
                'loss/running avg box size',
                'loss/running avg emptiness constraint',
                'loss/running avg tightness prior',
                'loss/running avg weighted sum',
            ],
            epoch_training_metrics=[
                'loss/box size',
                'loss/emptiness constraint',
                'loss/tightness prior',
                'loss/weighted sum',
                'metric/dice',
            ]
        )

        if experiment.comet is not None:
            experiment.comet.log_parameter('supervision', 'boxes')

            @trainer.on(Events.EPOCH_STARTED)
            def log_loss():
                """Log the current log barrier parameter to comet after each epoch"""
                experiment.comet.log_metric('log_barrier_t', loss_fn.log_barrier.t)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, step_loss, loss_fn=loss_fn)
        trainer.run(experiment.train_loader, max_epochs=args.max_epoch)


if __name__ == "__main__":
    main()
