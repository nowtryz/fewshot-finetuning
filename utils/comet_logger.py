from typing import Any, Callable, Union, Optional, List

from comet_ml import Experiment
from ignite.engine import Engine, Events
from ignite.handlers.base_logger import BaseLogger, BaseOutputHandler, BaseOptimizerParamsHandler


class CometLogger(BaseLogger):
    def __init__(self, *args, **kwargs) -> None:
        self.experiment = Experiment(*args, **kwargs)

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.experiment, attr)

    def close(self) -> None:
        self.experiment.end()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> Callable:
        return CometOutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> Callable:
        return CometOptimizerParamsHandler(*args, **kwargs)


class CometOutputHandler(BaseOutputHandler):
    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable[[Engine, Union[str, Events]], int]] = None,
        global_epoch_transform: Optional[Callable[[Engine], int]] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super().__init__(tag, metric_names, output_transform, global_step_transform, state_attributes)

        if global_epoch_transform is None:
            def global_epoch_transform(engine: Engine) -> int:
                return engine.state.epoch

        self.global_epoch_transform = global_epoch_transform

    def __call__(self, engine: Engine, logger: CometLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, CometLogger):
            raise TypeError(f"Handler '{self.__class__.__name__}' works only with CometLogger")

        global_step = self.global_step_transform(engine, event_name)
        global_epoch = self.global_epoch_transform(engine)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        metrics = self._setup_output_metrics_state_attrs(engine, log_text=True, key_tuple=False)
        logger.experiment.log_metrics(metrics, step=global_step, epoch=global_epoch)


class CometOptimizerParamsHandler(BaseOptimizerParamsHandler):
    def __call__(self, engine: Engine, logger: CometLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, CometLogger):
            raise TypeError(f"Handler '{self.__class__.__name__}' works only with CometLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        logger.experiment.log_parameters(params, step=global_step)
