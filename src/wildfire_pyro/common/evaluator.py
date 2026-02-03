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

        baseline_maes = []
        baseline_rmses = []
        model_maes = []
        model_rmses = []


        for _ in range(self.n_eval):
            obs, gt, baseline = self.env.get_bootstrap_observations(self.n_bootstrap)
            preds, _ = self.learner.predict(obs)

            preds_raw = self.env.to_raw_target(preds)
            gt_raw = self.env.to_raw_target(gt)
            baseline_raw = self.env.to_raw_target(baseline)




            mae_baseline = np.mean(np.abs(baseline_raw - gt_raw))
            rmse_baseline = np.sqrt(np.mean((baseline_raw - gt_raw) ** 2))
            baseline_maes.append(mae_baseline)
            baseline_rmses.append(rmse_baseline)

            mae_pred = np.mean(np.abs(preds_raw - gt_raw))
            rmse_pred = np.sqrt(np.mean((preds_raw - gt_raw) ** 2))
            model_maes.append(mae_pred)
            model_rmses.append(rmse_pred)





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
            baseline_mae_raw=float(np.mean(baseline_maes)),
            baseline_rmse_raw=float(np.mean(baseline_rmses)),
            model_mae_raw=float(np.mean(model_maes)),
            model_rmse_raw=float(np.mean(model_rmses)),
        )

    def _loss(self, pred, gt) -> float:
        p = torch.tensor(pred, dtype=torch.float32)
        g = torch.tensor(gt, dtype=torch.float32)
        return self.error_fn(p, g).item()
