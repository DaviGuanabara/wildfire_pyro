import pathlib
import numpy as np
from typing import Any, Dict, Optional, Tuple

from pyparsing import Iterable, Union
import io
import torch
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.components.observation_provider import StudentObservationProvider, TeacherObservationProvider
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from wildfire_pyro.wrappers.components import TeacherPredictionProvider, StudentPredictionProvider

class DistillationLearningManager(SupervisedLearningManager):
    """
    This manager performs online distillation where a lightweight teacher guides
    a student network. The student does not act in the environment. Instead,
    actions and targets come from the teacher.

    To change behavior (e.g., allow student to act), call:
        self.set_action_provider(StudentActionProvider(student_model, self.device))

    The current architecture supports:
    - Flexible substitution of action and target generators
    - Multiple observation shapes (student vs teacher)
    - Curriculum-style evolution of control over the environment
    """
    
    def __init__(
        self,
        student_nn: torch.nn.Module,
        teacher_nn: torch.nn.Module,
        environment,
        model_parameters: Dict[str, Any],
        logging_parameters: Dict[str, Any],
        runtime_parameters: Dict[str, Any],
        
        
    ):

        self._verify_environment(environment)

        super().__init__(
            neural_network=None,
            environment=environment,
            logging_parameters=logging_parameters,
            runtime_parameters=runtime_parameters,
            model_parameters=model_parameters,
        )

        self._set_neural_network(student_nn, teacher_nn)

    def _verify_environment(self, environment):
        if not hasattr(environment, "teacher_observation_space"):
            raise ValueError("Environment must have 'teacher_observation_space' attribute for DistillationLearningManager.")

        if not hasattr(environment, "student_observation_space"):
            raise ValueError("Environment must have 'student_observation_space' attribute for DistillationLearningManager.")

    def _set_neural_network(self, student_nn: torch.nn.Module, teacher_nn: torch.nn.Module,):
        """
        Sets the neural network for the learning manager.

        Args:
            neural_network (torch.nn.Module): The neural network to be set.
        """
        teacher_observation_space = self.environment.teacher_observation_space # type: ignore #
        student_observation_space = self.environment.student_observation_space # type: ignore #

        self.optimizer = torch.optim.Adam(
            student_nn.parameters(),
            lr=self.lr_fn(step=0, total_steps=1, loss=None),
        )

        self.label_provider = TeacherPredictionProvider(
            network=teacher_nn,
            observation_space=teacher_observation_space,
            device=self.device,
        )
        self.prediction_provider = StudentPredictionProvider(
            network=student_nn,
            observation_space=student_observation_space,
            device=self.device,
        )

        self.student_nn = student_nn
        self.teacher_nn = teacher_nn
        self.neural_network = student_nn

        self.prediction_obs_provider = StudentObservationProvider()
        self.label_obs_provider = TeacherObservationProvider()