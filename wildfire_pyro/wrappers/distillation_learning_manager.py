import pathlib
import numpy as np
from typing import Any, Dict, Optional, Tuple

from pyparsing import Iterable, Union
import io
import torch
from wildfire_pyro.environments.base_environment import BaseEnvironment
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

        self.label_provider = TeacherPredictionProvider(
            network=teacher_nn,
            observation_space=environment.teacher_observation_space,
            device=self.device,
        )
        self.prediction_provider = StudentPredictionProvider(
            network=student_nn,
            observation_space=environment.observation_space,
            device=self.device,
        )


    def _verify_environment(self, environment):
        if not hasattr(environment, "teacher_observation_space"):
            raise ValueError("Environment must have 'teacher_observation_space' attribute for DistillationLearningManager.")

        if not hasattr(environment, "student_observation_space"):
            raise ValueError("Environment must have 'student_observation_space' attribute for DistillationLearningManager.")

    def _set_neural_network(self, student_nn: torch.nn.Module,
        teacher_nn: torch.nn.Module,):
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

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Predict actions from observations.
        """
        raise NotImplementedError("Not defined for distillation.")
    
    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: Path to save the RL agent.
        :param exclude: Parameters to exclude from saving.
        :param include: Parameters to include (even if normally excluded).
        """
        path = pathlib.Path(path) if isinstance(path, str) else path

        # Copy instance attributes
        data = self.__dict__.copy()

        # Default parameters to exclude from saving
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)

        exclude.update(self._excluded_save_params())

        # Ensure explicitly included parameters are not excluded
        if include is not None:
            exclude.difference_update(include)

        # Handle PyTorch parameters
        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        pytorch_variables = {
            name: self.recursive_getattr(self, name) for name in torch_variable_names
        }

        # Remove excluded attributes
        for param_name in exclude:
            data.pop(param_name, None)

        # Get state dictionaries
        params_to_save = self.get_parameters()

        # Save everything to a zip file
        self.save_to_zip_file(
            path, data=data, params=params_to_save, pytorch_variables=pytorch_variables
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve model parameters.
        """
        return {"student_neural_network": self.student_nn.state_dict(), "teacher_neural_network": self.teacher_nn.state_dict()}