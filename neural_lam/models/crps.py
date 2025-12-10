# Standard library
import math

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr

# First-party
from neural_lam import metrics, vis
from neural_lam.models.EDM import EDM
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class CRPS(EDM):
    """
    Continuous Ranked Probability Score (CRPS) Model.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
    ):
        super().__init__(args, config, datastore)

        if args.backbone_model == "graph_diff":
            print("Using GraphDiff")
            self.model = GraphDiff(args, config, datastore)
        else:
            raise NotImplementedError(
                f"Unknown backbone model: {args.backbone_model}"
            )

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        next_state: (B, N_grid, d_state),
            predicted weather state X_{t+1} at time t+1
        """
        input_grid = torch.cat((prev_state, prev_prev_state, forcing),
                               dim=-1)  # (B, N_grid, d_input)

        z = torch.randn(
            prev_state.shape[0], self.model.noise_dim, device=prev_state.device)

        next_state = self.model(
            input_grid, noise_level=z, cond=None)  # (B, N_grid, d_f)

        # Add residual if needed
        if self.pred_residual:
            next_state = (next_state * self.diff_std) + \
                self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        return next_state

    def training_step(self, batch):
        """
        Train on single batch
        batch consists of:
        init_states: (B, 2, num_interior_nodes, d_features)
        target_states: (B, pred_steps, num_interior_nodes, d_features)
        forcing: (B, pred_steps, num_interior_nodes, d_forcing),
        boundary_forcing:
            (B, pred_steps, num_boundary_nodes, d_boundary_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (init_states, target_states, forcing, _) = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing,
            target_states,
            # NOTE: We always use ensemble size 2 for training (following FGN)
            2,
        )

        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            average_grid=False,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        # prediction: (B, pred_steps, num_interior_nodes, d_f)
        # pred_std: (B, pred_steps, num_interior_nodes, d_f) or (d_f,)
        # loss_batch: (B, pred_steps)
        loss_batch = metrics.mask_and_reduce_metric(
            crps_batch / self.per_var_std, mask=self.interior_mask_bool, average_grid=True, sum_vars=True)

        # Compute loss - mean over unrolled times and batch
        batch_loss = torch.mean(loss_batch)

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss
