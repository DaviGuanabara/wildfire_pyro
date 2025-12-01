import numpy as np
from wildfire_pyro.environments.iowa.components.adapter_params import AdapterParams
from wildfire_pyro.environments.iowa.components.metadata import Metadata
from wildfire_pyro.environments.iowa.parametric_environment import ParametricEnvironment



class IowaEnvironment(ParametricEnvironment):
    def __init__(self, data_path, verbose: bool = False, baseline_type="mean_neighbor"):

        metadata = self._metadata()
        params = self._params(verbose)

        super().__init__(data_path=data_path, metadata=metadata, params=params, baseline_type=baseline_type)

    def _params(self, verbose: bool) -> AdapterParams:
        return AdapterParams(
            min_neighborhood_size=2,
            max_neighborhood_size=10,
            max_delta_distance=1e9,
            max_delta_time=10.0,
            verbose=verbose,
        )

    def _metadata(self) -> Metadata:

        return Metadata(
            time="valid",  # coluna de tempo
            position=["Latitude1", "Longitude1"],  # colunas espaciais
            id="station",  # coluna de identificação
            features=[
                "in_high", "in_low",  # temperature
                "in_rh_min", "in_rh", "in_rh_max",  # relative humidity min, avg, max
                "in_solar_mj",  # solar radiation

                "in_precip",  # preciptation
                "in_speed",  # wind speed
                # A sudden, brief increase in wind speed, typically lasting 2–5 seconds, above the mean wind speed.
                "in_gust",
                "in_et",  # evapotranspiration
                "Elevation [m]",  # elevation
            ],
            # , "out_lwmwet_2_tot"]  # colunas alvo
            target=["out_lwmwet_1_tot"],
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV de treino
    data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\train.csv"
    data_path_mac = "/Users/Davi/Documents/GitHub/wildfire_workspace/wildfire/examples/iowa_soil/data/train.csv"
    data_path_windows = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\examples\\iowa_soil\\data\\processed\\tidy_isusm_stations.csv"
    data_path = data_path_mac
    # Instancia o ambiente
    env = IowaEnvironment(data_path=data_path_windows, verbose=True)

    # Reset do ambiente
    obs, info = env.reset()

    print("\n=== Reset Environment ===")
    print("Observation keys:", list(obs.keys()))
    print("Neighbors shape:", obs["neighbors"].shape)
    print("Mask shape:", obs["mask"].shape)
    print("Ground Truth:", info["ground_truth"])

    # Executa alguns passos
    print("\n=== Step Environment ===")
    for step_id in range(3):

        obs, reward, terminated, truncated, info = env.step()

        print(f"\nStep {step_id + 1}")
        print("Observation[neighbors] shape:", obs["neighbors"].shape)
        print("Observation[mask] shape:", obs["mask"].shape, obs["mask"])
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Ground Truth:", info["ground_truth"])
        print("feature_names:", info["feature_names"])

    bootstrap_obs, _ = env.get_bootstrap_observations(n_bootstrap=5)
    baseline_pred = env.baseline(bootstrap_observations=bootstrap_obs)

    print("\n=== Bootstrap Baseline Predictions ===")
    print("Baseline predictions:", baseline_pred)
