# iowa_experiment.py

from typing import Tuple
from dataclasses import asdict

from wildfire_pyro.common.callbacks import CallbackList, TrainLoggingCallback
from wildfire_pyro.common.evaluator import BootstrapEvaluator
from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.common.seed_manager import configure_seed_manager
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.factories.learner_factory import create_deep_set_learner
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from wildfire_pyro.helpers.parameters import RunParameters


class IowaEnvironmentExperiment:
    def __init__(self, config: RunParameters):
        self.config = config

    def setup(self):
        # Single source of randomness for the whole experiment
        self.seed_manager = configure_seed_manager(
            global_seed=self.config.runtime_parameters.GLOBAL_SEED
        )

        # Environments
        self.train_env = IowaEnvironment(
            data_path=self.config.data_parameters.train_path,
            verbose=self.config.runtime_parameters.verbose,
            seed=self.seed_manager.get_seed("train_env"),
        )

        self.test_env = IowaEnvironment(
            data_path=self.config.data_parameters.test_path,
            scaler=self.train_env.get_fitted_scaler(),
            verbose=self.config.runtime_parameters.verbose,
            seed=self.seed_manager.get_seed("test_env"),
        )

        # Learner
        self.learner = create_deep_set_learner(
            env=self.train_env,
            model_parameters=self.config.model_parameters,
            logging_parameters=self.config.logging_parameters,
            runtime_parameters=self.config.runtime_parameters,
            seed=self.seed_manager.get_seed("learner"),
        )

    def _train(self) -> SupervisedLearningManager:
        train_callback = TrainLoggingCallback(
            log_freq=self.config.training_parameters.log_frequency,
            verbose=self.config.runtime_parameters.verbose,
        )

        callbacks = CallbackList([train_callback])

        # self.train_env.reset()

        self.learner.learn(
            total_timesteps=self.config.training_parameters.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        return self.learner

    def _test(self) -> EvaluationMetrics:
        self.test_env.reset()

        evaluator = BootstrapEvaluator(
            environment=self.test_env,
            learner=self.learner,
            n_eval=self.config.test_parameters.n_eval,
            n_bootstrap=self.config.test_parameters.n_bootstrap,
        )

        return evaluator.evaluate(self.seed_manager.get_seed("evaluator"))

    def teardown(self):
        self.train_env.close()
        self.test_env.close()

    def run(self) -> Tuple[SupervisedLearningManager, EvaluationMetrics]:
        try:
            self.setup()
            model = self._train()
            metrics = self._test()
            return model, metrics
        finally:
            self.teardown()
