"""Shared training callbacks."""

import logging

import numpy as np
import transformers

logger = logging.getLogger(__name__)


class NaNLossCallback(transformers.TrainerCallback):
    """Halt training after too many consecutive NaN loss steps."""

    def __init__(self, max_consecutive: int = 5):
        self.max_consecutive = max_consecutive
        self._consecutive_nan = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            self._consecutive_nan += 1
            logger.warning(
                "NaN/Inf loss detected (step %d, %d consecutive)",
                state.global_step,
                self._consecutive_nan,
            )
            if self._consecutive_nan >= self.max_consecutive:
                logger.error(
                    "Halting training: %d consecutive NaN loss steps",
                    self._consecutive_nan,
                )
                control.should_training_stop = True
        else:
            self._consecutive_nan = 0
