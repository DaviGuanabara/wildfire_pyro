from typing import Callable, Optional
import numpy as np
import torch
from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.base_learning_manager import BaseLearningManager


class BootstrapEvaluator:
    def __init__(
        self,
        environment: BaseEnvironment,
        learner: BaseLearningManager,
        n_eval: int,
        n_bootstrap: int,
        error_function: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.env = environment
        self.learner = learner
        self.n_eval = n_eval
        self.n_bootstrap = n_bootstrap
        self.error_fn = error_function or torch.nn.MSELoss()
        self.seed = seed

    def evaluate(self) -> EvaluationMetrics:
        self.env.reset(self.seed)

        nn_errors = []
        baseline_errors = []
        comparisons = []

        for _ in range(self.n_eval):
            obs, gt, baseline = self.env.get_bootstrap_observations(self.n_bootstrap)
            preds, _ = self.learner.predict(obs)

            nn_err = self._loss(preds, gt)
            base_err = self._loss(baseline, gt)

            nn_errors.append(nn_err)
            baseline_errors.append(base_err)
            comparisons.append(int(nn_err < base_err))

            self.env.step()

        return EvaluationMetrics(
            model_error=float(np.mean(nn_errors)),
            model_std=float(np.std(nn_errors)),
            baseline_error=float(np.mean(baseline_errors)),
            baseline_std=float(np.std(baseline_errors)),
            model_win_rate_over_baseline=float(np.mean(comparisons)),
        )

    def _loss(self, pred, gt) -> float:
        p = torch.tensor(pred, dtype=torch.float32)
        g = torch.tensor(gt, dtype=torch.float32)
        return self.error_fn(p, g).item()
