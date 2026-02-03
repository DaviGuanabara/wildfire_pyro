from wildfire_pyro.common.evaluator import BootstrapEvaluator
from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager

from wildfire_pyro.factories.learner_factory import RunConfig


def evaluate(
    eval_environment: IowaEnvironment,
    deep_set_learner: SupervisedLearningManager,
    config: RunConfig,
) -> EvaluationMetrics:

    seed = config.runtime_parameters.seed

    evaluator: BootstrapEvaluator = BootstrapEvaluator(
        environment=eval_environment,
        learner=deep_set_learner,
        n_eval=config.test_parameters.n_eval,
        n_bootstrap=config.test_parameters.n_bootstrap,
        seed=seed,
    )
    metrics: EvaluationMetrics = evaluator.evaluate()
    eval_environment.close()

    return metrics

