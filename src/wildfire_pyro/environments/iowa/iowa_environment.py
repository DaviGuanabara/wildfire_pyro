import numpy as np
from wildfire_pyro.environments.iowa.components.adapter_params import AdapterParams
from wildfire_pyro.environments.iowa.components.meta_data import Metadata
from wildfire_pyro.environments.iowa.adaptative_environment import (
    AdaptativeEnvironment,
)


class IowaEnvironment(AdaptativeEnvironment):
    def __init__(self, data_path, verbose: bool = False):

        metadata = self._metadata()
        params = self._params(verbose)

        super().__init__(
            data_path=data_path, metadata=metadata, params=params
        )

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
            position=["Latitude1", "Longitude1", "Elevation [m]"],  # colunas espaciais
            id="station",  # coluna de identificação
            exclude=[
                "out_lwmv_1",
                "out_lwmv_2",
                "out_lwmdry_1_tot",
                "out_lwmcon_1_tot",
                "out_lwmdry_2_tot",
                "out_lwmcon_2_tot",
                "out_lwmwet_2_tot",  # colunas a excluir
                "ID",
                "Archive Begins",
                "Archive Ends",
                "IEM Network",
                "Attributes",
                "Station Name",
            ],
            # , "out_lwmwet_2_tot"]  # colunas alvo
            target=["out_lwmwet_1_tot"],
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # ⚠️ Preencha com o caminho real do seu CSV de treino
    data_path = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\wildfire_pyro\\examples\\iowa_soil\\data\\train.csv"

    # Instancia o ambiente
    env = IowaEnvironment(data_path=data_path, verbose=True)

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
