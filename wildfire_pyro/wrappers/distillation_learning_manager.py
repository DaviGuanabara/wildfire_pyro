import numpy as np
from typing import Any, Dict

import torch
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from .components import TeacherTargetProvider
from .components import StudentActionProvider, TeacherActionProvider

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
        student: torch.nn.Module,
        teacher_nn: torch.nn.Module,
        environment: BaseEnvironment,
        model_parameters: Dict[str, Any],
        logging_parameters: Dict[str, Any],
        runtime_parameters: Dict[str, Any],
        
        
    ):
        super().__init__(
            neural_network=student,
            environment=environment,
            logging_parameters=logging_parameters,
            runtime_parameters=runtime_parameters,
            model_parameters=model_parameters,
        )
        
        self.target_provider = TeacherTargetProvider(
            teacher=teacher_nn,
            observation_space=environment.teacher_observation_space,
            device=self.device,
        )
        
        self.action_provider = TeacherActionProvider(
            teacher=teacher_nn, device=self.device)
