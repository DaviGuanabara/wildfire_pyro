from typing import Any

import numpy as np
from gymnasium import spaces

class BaseObservationProvider:

    def get_observation(self, observation, info):
        raise NotImplementedError


class TeacherObservationProvider(BaseObservationProvider) :
    def get_observation(self, observation, info) -> spaces.Space:
        # Extract the teacher's observation from the state
        return info["teacher_observation"]

class StudentObservationProvider(BaseObservationProvider):
    def get_observation(self, observation, info) -> spaces.Space:
        # Extract the student's observation from the state
        return info["student_observation"]

class PredictionObservationProvider(BaseObservationProvider):
    def get_observation(self, observation, info) -> spaces.Space:
        # Extract the default observation from the state
        return observation

class LabelObservationProvider(BaseObservationProvider):
    def get_observation(self, observation, info) -> spaces.Space:
        # Extract the label observation from the state
        return info["ground_truth"]