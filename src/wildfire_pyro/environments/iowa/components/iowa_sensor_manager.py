from typing import Tuple, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)


class IOWASensorManager:

    def __init__(self, data_path, verbose: bool = False):
        """
        Initialize the SensorManager with the dataset.

        Args:
            data_path (str): Path to the dataset file.
        """

        self.verbose = verbose
        self.data = pd.read_csv(data_path)

        self.init_tags()

    def init_tags(self):

        """
        Here is to say what is temporal, spatial, target, exclude, etc.
        what is no directly cited, is considered feature.
        """

        self.tags = {
            "exclude": "id",
            "target": ["high", "low"],
            "features":{
                "temporal": "data",
                "spatial": ["Latitude1", "Longitude1", "Elevation [m]"]
            }
        }

    def _limits(self):
        """
        Define the observation and action spaces.

        Returns:
            Tuple[Any, Any]: Observation space and action space.
        """
        data = self.data
        data = data.drop(columns=[self.tags["exclude"]])
        limits = data.agg(["min", "max"]).T

        return limits

if __name__ == "__main__":

    iowa_sensor = IOWASensorManager(
        data_path="C:\\Users\\davi_\\Documents\\GitHub\\wildfire_project\\workspace\\wildfire_pyro\\examples\\iowa_soil\\data\\ISU_Soil_Moisture_Network\\dataset_preprocessed", verbose=True
    )
    iowa_sensor._limits()