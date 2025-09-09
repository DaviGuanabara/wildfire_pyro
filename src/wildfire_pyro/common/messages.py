from dataclasses import dataclass, field
import numpy as np




@dataclass
class EvaluationMetrics:
    model_error: float
    model_std: float
    baseline_error: float = np.nan
    baseline_std: float = np.nan
    model_win_rate_over_baseline: float = np.nan
    context_type: str = field(default="EvaluationMetrics", init=False, repr=False)

    def has_baseline(self) -> bool:
        return not np.isnan(self.baseline_error)
