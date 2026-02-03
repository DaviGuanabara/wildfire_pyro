from wildfire_pyro.common.callbacks import CallbackList, TrainLoggingCallback
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment

from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from wildfire_pyro.factories.learner_factory import RunConfig


def train(
    train_environment: IowaEnvironment,
    deep_set_learner: SupervisedLearningManager,
    config: RunConfig,
) -> SupervisedLearningManager:

    seed = config.runtime_parameters.seed
    training_parameters = config.training_parameters

    train_callback = TrainLoggingCallback(
        log_freq=training_parameters.log_frequency, verbose=config.runtime_parameters.verbose)


    callbacks = CallbackList([train_callback])
    train_environment.reset(seed)


    deep_set_learner.learn(
        total_timesteps=training_parameters.total_timesteps, callback=callbacks, progress_bar=True
    )

    train_environment.close()

    return deep_set_learner

