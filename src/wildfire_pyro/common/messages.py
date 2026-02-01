from dataclasses import dataclass, field
from typing import Optional
import numpy as np




@dataclass
class EvaluationMetrics:
    model_error: float
    model_std: float
    baseline_error: float = np.nan
    baseline_std: float = np.nan
    model_win_rate_over_baseline: float = np.nan
    context_type: str = field(default="EvaluationMetrics", init=False, repr=False)

    # 🔹 NOVO — métricas interpretáveis (raw space)
    model_mae_raw: Optional[float] = None
    model_rmse_raw: Optional[float] = None
    baseline_mae_raw: Optional[float] = None
    baseline_rmse_raw: Optional[float] = None
    
    def has_baseline(self) -> bool:
        return not np.isnan(self.baseline_error)
