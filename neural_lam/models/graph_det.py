# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import torch
import wandb

# First-party
from neural_lam import metrics
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class GraphDET(ARModel):
    """
    SeaCast with learnable layernorm.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            **kwargs,
        )

        if args.backbone_model == "graph_diff":
            print("Using GraphDiff")
            self.model = GraphDiff(
                args,
                config,
                datastore,
                datastore_boundary,
                datastore_atmosphere,
            )
        else:
            raise NotImplementedError(
                f"Unknown backbone model: {args.backbone_model}"
            )

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)


        Returns:
        next_state: (B, N_grid, d_state),
            predicted weather state X_{t+1} at time t+1
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)
        # (B, N_grid, d_input)

        z = torch.ones(
            prev_state.shape[0],
            self.model.noise_dim,
            device=prev_state.device) * self.args.deterministic_noise_value

        next_state = self.model(
            input_grid,
            noise_level=z,
            cond=None,
            boundary_forcing=boundary_forcing,
            atmosphere_forcing=atmosphere_forcing,
        )  # (B, N_grid, d_f)

        # Add residual if needed
        if self.args.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        return next_state, self.per_var_std
