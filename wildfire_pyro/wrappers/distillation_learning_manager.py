import numpy as np
from typing import Any, Dict

import torch
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager
from .components import TeacherTargetProvider

# TODO: Devo alterar mais em baixo a questao de ter
# muktiplas observacoes. a obs principal vem do retorno do step
# e ela refere-se ao student. já a segunda obs, 
# deve estar dentro do info (wue restorna do step do ambiente) e é direcionado ao teacher
# algo como info[teacher_observation]. 
# isso é necessario para o caso do student receber un espaco de observacao maior
# do wue p teacher (ou ao menos diferente)
# assim tem wue tambem lidar com o observation shape do teacher. 
class DistillationLearningManager(SupervisedLearningManager):
    def __init__(
        self,
        neural_network: torch.nn.Module,
        environment: BaseEnvironment,
        logging_parameters: Dict[str, Any],
        runtime_parameters: Dict[str, Any],
        model_parameters: Dict[str, Any],
        teacher: torch.nn.Module,
        target_info_key: str = "ground_truth",
    ):
        super().__init__(
            neural_network=neural_network,
            environment=environment,
            logging_parameters=logging_parameters,
            runtime_parameters=runtime_parameters,
            model_parameters=model_parameters,
            target_info_key=target_info_key,
            target_provider=TeacherTargetProvider(
                teacher=teacher,
                target_info_key=target_info_key,
                input_shape=environment.observation_space.shape,
                device=runtime_parameters.get("device", "cpu"),
            ),
        )
