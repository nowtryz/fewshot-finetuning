from functools import partial

import comet_ml  # noqa
import torch
from ignite.engine import create_supervised_trainer
from ignite.metrics import Loss, EpochMetric, Average

from pretrain.datasets.tranformation_pipeline import make_preprocessing_transforms, make_augmentation_transforms, \
    make_validation_transforms
from utils.engine import setup_experiment, setup_trainer, setup_evaluator, setup_logger
from utils.losses import BinaryDice3D
from utils.misc import capture_duration, monitor_resources
from utils.parser import make_main_parser


def prepare_batch(batch, device, non_blocking):
    x, y = batch["image"].to(device, non_blocking=non_blocking), batch["label"].to(device, non_blocking=non_blocking)
    annotation_mask = batch['annotation_mask'].to(device, non_blocking=non_blocking)
    return x, (y, annotation_mask)


def apply_model(model, x):
    logit_map = model(x)  # Forward
    return torch.sigmoid(logit_map)  # Activation


def loss_fn(criterion, pred, gt):
    y, annotation_mask = gt
    return criterion(pred, y, annotation_mask=annotation_mask)


def validation_output_transform(x, y, y_pred):
    """
    Process output from the model, retrieve the annotation mask and move it to ``criterion_kwargs`` for the `Loss`
    metric.
    """
    y, mask = y
    return y_pred, y, {"annotation_mask": mask}


@capture_duration()
@monitor_resources()
def main():
    parser = make_main_parser()
    args = parser.parse_args()

    with setup_experiment(
        args=args,
        preprocessing_transforms=make_preprocessing_transforms(args),
        augmentation_transforms=make_augmentation_transforms(args),
        validation_transforms=make_validation_transforms(args)
    ) as experiment:
        criterion = BinaryDice3D().to(experiment.device)
        trainer = create_supervised_trainer(
            model=experiment.model,
            model_fn=apply_model,
            optimizer=experiment.optimizer,
            loss_fn=partial(loss_fn, criterion),
            device=experiment.device,
            prepare_batch=prepare_batch,
            # TODO would require to customise learning rate scheduler to use accumulation
            # gradient_accumulation_steps=4,
        )

        setup_trainer(trainer, output_names=['loss/dice'])
        evaluator = setup_evaluator(
            trainer,
            prepare_batch=prepare_batch,
            output_transform=validation_output_transform,
            metrics={
                "metric/dice": Loss(criterion)
            }
        )

        # Average loss metric
        Average(device=experiment.device).attach(trainer, 'metric/dice')

        if experiment.comet is not None:
            experiment.comet.log_parameter('supervision', 'segmentations')

        setup_logger(trainer, evaluator, training_metrics=['loss/dice'], epoch_training_metrics=['metric/dice'])
        trainer.run(experiment.train_loader, max_epochs=args.max_epoch)


if __name__ == "__main__":
    main()
