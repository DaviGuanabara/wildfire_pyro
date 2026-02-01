from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager

from wildfire.experiments.iowa_soil.train import train
from wildfire.experiments.iowa_soil.evaluate import evaluate

from wildfire.experiments.iowa_soil.config import RunConfig


def run(
    train_environment: IowaEnvironment,
    eval_environment: IowaEnvironment,
    deep_set_learner: SupervisedLearningManager,
    config: RunConfig,
) -> tuple[SupervisedLearningManager, EvaluationMetrics]:

    trained = train(
        train_environment=train_environment,
        deep_set_learner=deep_set_learner,
        config=config,
    )

    metrics = evaluate(
        eval_environment=eval_environment,
        deep_set_learner=trained,
        config=config,
    )

    return trained, metrics
