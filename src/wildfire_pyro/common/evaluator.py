from typing import List, Optional, Tuple
import numpy as np

from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.base_learning_manager import BaseLearningManager
from wildfire_pyro.common.seed_manager import get_seed_manager


class BootstrapEvaluator:

    """
    Statistical Evaluation Methodology
    ----------------------------------

    This evaluator implements a hierarchical (nested) bootstrap procedure to
    quantify both predictive performance and uncertainty in contextual
    regression problems.

    Let Y be the true target variable and Ŷ the model prediction. The objective
    is to estimate the expected absolute prediction error:

        E[ |Ŷ − Y| ]

    when predictions depend on a stochastic and partially observed context
    (e.g., neighboring samples, sensors, or spatial-temporal neighborhoods).

    --------------------
    Sources of Uncertainty
    --------------------

    Two distinct sources of uncertainty are explicitly modeled:

    (1) Conditional (local) uncertainty:
        For a fixed prediction target (pivot), the available contextual
        information is stochastic. Different resamplings of the context induce
        variability in the model prediction even when the ground truth is fixed.

        This uncertainty is quantified via a local bootstrap, estimating the
        conditional error distribution:

            |Ŷ − Y| | Target

    (2) Population-level variability:
        Different pivots correspond to different prediction tasks with varying
        difficulty, noise characteristics, and contextual richness.

        This variability is captured by evaluating multiple pivots sampled from
        the dataset.

    --------------------
    Hierarchical Bootstrap Structure
    --------------------

    Inner loop (local bootstrap):
        - Target is fixed
        - Context is resampled multiple times
        - The environment state does NOT advance
        - Produces a distribution of errors per pivot

    Outer loop (population evaluation):
        - Target changes at each iteration
        - The environment state advances
        - Produces a sample of pivot-level mean errors

    --------------------
    Estimated Quantities
    --------------------

    From the population of pivot-level mean errors, the following quantities are
    estimated:

    • Mean Absolute Error (MAE):
        An empirical estimate of the expected absolute prediction error.

    • Standard deviation across pivots:
        Measures heterogeneity of the prediction problem and robustness of the
        model across different targets.



    --------------------
    Interpretation
    --------------------

    The reported mean MAE reflects predictive accuracy.
    The standard deviation reflects task heterogeneity.

    Importantly, the confidence interval does NOT describe the variability of
    individual predictions, but the uncertainty of the estimated mean error
    itself.

    --------------------
    Scope and Limitations
    --------------------

    This evaluator does not estimate:
    - Aleatoric uncertainty of the data-generating process
    - Epistemic uncertainty over model parameters
    - Predictive uncertainty intervals for individual samples

    These aspects are intentionally excluded to preserve interpretability and
    computational tractability.
    """


    def __init__(
        self,
        environment: BaseEnvironment,
        learner: BaseLearningManager,
        n_eval: int,
        n_bootstrap: int,
        seed: int,
    ):
        self.env = environment
        self.learner = learner
        self.n_eval = n_eval
        self.n_bootstrap = n_bootstrap
        self.seed = seed


        

    def _local_bootstrap_errors(
            self,
        ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Performs LOCAL bootstrap evaluation for a SINGLE pivot (fixed target).

        This function estimates the model uncertainty conditioned on the SAME
        underlying target. For that reason, **env.step() MUST NOT be called here**.

        Each bootstrap iteration:
        - Resamples neighbors/context around the same pivot
        - Produces a prediction
        - Computes MAE and RMSE against the fixed ground truth

        Returns
        -------
        model_maes : list of float
            MAE values of the model for each bootstrap resampling
        baseline_maes : list of float
            MAE values of the baseline for each bootstrap resampling
        model_rmses : list of float
            RMSE values of the model (diagnostic)
        baseline_rmses : list of float
            RMSE values of the baseline (diagnostic)
        """

        obs, gt, baseline = self.env.get_bootstrap_observations(
            self.n_bootstrap)

        preds, _ = self.learner.predict(obs)

        preds_raw = self.env.to_raw_target(preds)
        gt_raw = self.env.to_raw_target(gt)
        baseline_raw = self.env.to_raw_target(baseline)

        # shape: (n_bootstrap, ...)
        model_maes = np.mean(np.abs(preds_raw - gt_raw), axis=1)
        baseline_maes = np.mean(np.abs(baseline_raw - gt_raw), axis=1)

        model_rmses = np.sqrt(np.mean((preds_raw - gt_raw) ** 2, axis=1))
        baseline_rmses = np.sqrt(np.mean((baseline_raw - gt_raw) ** 2, axis=1))

        return (
            model_maes.tolist(),
            baseline_maes.tolist(),
            model_rmses.tolist(),
            baseline_rmses.tolist(),
        )

    def _evaluate_single_pivot(
        self,
    ) -> Tuple[float, float, float, float, int]:
        """
        Evaluates a SINGLE pivot (fixed target) by aggregating its local bootstrap.

        This function:
        - Calls the local bootstrap routine
        - Aggregates the bootstrap samples into a single MAE/RMSE estimate
        - Determines whether the model beats the baseline for this pivot

        IMPORTANT:
        - The pivot is still FIXED here
        - env.step() MUST NOT be called in this function

        Returns
        -------
        model_mae : float
            Mean MAE of the model over local bootstrap samples
        baseline_mae : float
            Mean MAE of the baseline over local bootstrap samples
        model_rmse : float
            Mean RMSE of the model (diagnostic)
        baseline_rmse : float
            Mean RMSE of the baseline (diagnostic)
        win : int
            1 if model outperforms baseline for this pivot, else 0
        """

        (
            model_maes,
            baseline_maes,
            model_rmses,
            baseline_rmses,
        ) = self._local_bootstrap_errors()

        model_mae = float(np.mean(model_maes))
        baseline_mae = float(np.mean(baseline_maes))

        model_rmse = float(np.mean(model_rmses))
        baseline_rmse = float(np.mean(baseline_rmses))

        win = int(model_mae < baseline_mae)

        return model_mae, baseline_mae, model_rmse, baseline_rmse, win

    def evaluate(self) -> EvaluationMetrics:
        """
        Performs GLOBAL evaluation over multiple pivots.

        In addition to mean and standard deviation across pivots,
        this method estimates a bootstrap confidence interval (CI)
        for the mean MAE, reflecting the uncertainty of the estimated
        expected error.
        """

        self.env.reset(self.seed)

        model_maes = []
        baseline_maes = []

        model_rmses = []
        baseline_rmses = []

        wins = []

        # ===============================
        # Outer loop: population of pivots
        # ===============================
        for _ in range(self.n_eval):
            (
                model_mae,
                baseline_mae,
                model_rmse,
                baseline_rmse,
                win,
            ) = self._evaluate_single_pivot()

            model_maes.append(model_mae)
            baseline_maes.append(baseline_mae)

            model_rmses.append(model_rmse)
            baseline_rmses.append(baseline_rmse)

            wins.append(win)

            self.env.step()

        model_maes = np.asarray(model_maes)
        baseline_maes = np.asarray(baseline_maes)



        return EvaluationMetrics(
            # Central tendency
            model_mae_mean=float(np.mean(model_maes)),
            model_mae_std=float(np.std(model_maes, ddof=1)),

            baseline_mae_mean=float(np.mean(baseline_maes)),
            baseline_mae_std=float(np.std(baseline_maes, ddof=1)),

            # Diagnostics
            model_rmse_mean=float(np.mean(model_rmses)),
            baseline_rmse_mean=float(np.mean(baseline_rmses)),

            win_rate_over_baseline=float(np.mean(wins)),
        )
