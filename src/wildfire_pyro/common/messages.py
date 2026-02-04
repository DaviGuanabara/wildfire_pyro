from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvaluationMetrics:

    model_mae_mean: float
    model_mae_std: float

    baseline_mae_mean: float
    baseline_mae_std: float

    win_rate_over_baseline: float

    model_rmse_mean: float
    baseline_rmse_mean: float