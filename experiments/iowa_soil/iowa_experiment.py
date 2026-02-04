# iowa_experiment.py

from typing import Tuple
from dataclasses import asdict

from wildfire_pyro.common.callbacks import CallbackList, TrainLoggingCallback
from wildfire_pyro.common.evaluator import BootstrapEvaluator
from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.common.seed_manager import configure_seed_manager
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from parameters import RunParameters


class IowaEnvironmentExperiment:
    def __init__(self, config: RunParameters):
        self.config = config

    def setup(self):
        # Single source of randomness for the whole experiment
        self.seed_manager = configure_seed_manager(
            self.config.runtime_parameters.seed
        )

        # Environments
        self.train_env = IowaEnvironment(
            data_path=self.config.data_parameters.train_path,
            verbose=self.config.runtime_parameters.verbose,
        )

        self.test_env = IowaEnvironment(
            data_path=self.config.data_parameters.test_path,
            scaler=self.train_env.get_fitted_scaler(),
            verbose=self.config.runtime_parameters.verbose,
        )

        # Learner
        net = DeepSetAttentionNet(
            observation_space=self.train_env.observation_space,
            action_space=self.train_env.action_space,
            hidden_dim=self.config.model_parameters.hidden,
            prob=self.config.model_parameters.dropout_prob,
        ).to(self.config.runtime_parameters.device)

        self.learner = SupervisedLearningManager(
            neural_network=net,
            environment=self.train_env,
            logging_parameters=asdict(self.config.logging_parameters),
            runtime_parameters=asdict(self.config.runtime_parameters),
            model_parameters=asdict(self.config.model_parameters),
        )

    def _train(self) -> SupervisedLearningManager:
        train_callback = TrainLoggingCallback(
            log_freq=self.config.training_parameters.log_frequency,
            verbose=self.config.runtime_parameters.verbose,
        )

        callbacks = CallbackList([train_callback])

        self.train_env.reset(
            self.seed_manager.get_seed("train")
        )

        self.learner.learn(
            total_timesteps=self.config.training_parameters.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        return self.learner

    def _test(self) -> EvaluationMetrics:
        self.test_env.reset(
            self.seed_manager.get_seed("test")
        )

        evaluator = BootstrapEvaluator(
            environment=self.test_env,
            learner=self.learner,
            n_eval=self.config.test_parameters.n_eval,
            n_bootstrap=self.config.test_parameters.n_bootstrap,
            seed=self.seed_manager.get_seed("bootstrap_evaluator"),
        )

        return evaluator.evaluate()

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
