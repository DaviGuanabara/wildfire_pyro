import numpy as np
from wildfire_pyro.common.baselines.BaselineStrategy import BaselineStrategy
from wildfire_pyro.common.baselines.BaselineRegistry import register_baseline


@register_baseline("mean_neighbor")
class MeanNeighborBaseline(BaselineStrategy):

    def set_schema(self, neighbor_schema):
        """Environment injects schema once."""
        self.neighbor_schema = neighbor_schema

    def predict(self, observations):
        preds = []

        target_name = self.action_space.shape[0] == 1 \
            and self.neighbor_schema.target_cols[0]

        target_idx = self.neighbor_schema.index_map[target_name]

        for obs in observations:
            neigh = obs["neighbors"]          # (M, F)
            mask = obs["mask"].astype(bool)  # (M,)

            valid = neigh[mask, target_idx]

            if valid.size == 0 or np.isnan(valid).all():
                pred = np.zeros(self.action_space.shape, dtype=np.float32)
            else:
                pred = np.nanmean(valid).reshape(self.action_space.shape)

            # Already scaled — no re-scaling needed
            preds.append(pred.astype(np.float32))

        return np.vstack(preds)
