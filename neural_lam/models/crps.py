# Standard library
import math
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr

# First-party
from neural_lam import metrics, vis
from neural_lam.models.ar_prob_model import ARProbModel
from neural_lam.models.EDM import EDM
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class CRPS(ARProbModel):
    """
    Continuous Ranked Probability Score (CRPS) Model.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
    ):
        super().__init__(
            args, config, datastore, datastore_boundary, datastore_atmosphere
        )
        self.val_metrics.update(
            {
                "ens_crps": [],
            }
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
        input_grid = torch.cat(
            (prev_state, prev_prev_state, forcing), dim=-1
        )  # (B, N_grid, d_input)

        z = torch.randn(
            prev_state.shape[0], self.model.noise_dim, device=prev_state.device
        )

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
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            ensemble_size=2,
        )

        crps_batch = metrics.afcrps_ens(
            trajectories,
            target_states,
            None,
            average_grid=False,
            sum_vars=False,
            alpha=self.args.crps_alpha,
        )  # (B, pred_steps, d_f)
        # prediction: (B, pred_steps, num_interior_nodes, d_f)
        # pred_std: (B, pred_steps, num_interior_nodes, d_f) or (d_f,)
        # loss_batch: (B, pred_steps)
        loss_batch = metrics.mask_and_reduce_metric(
            crps_batch / self.per_var_std,
            mask=self.interior_mask_bool,
            average_grid=True,
            sum_vars=True,
        )

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

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        super().validation_step(batch, *args)
        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_step(batch)
        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            mask=self.interior_mask_bool,
            average_grid=True,
            sum_vars=False,
        )
        self.val_metrics["ens_crps"].append(crps_batch)

        if self.trainer.is_global_zero and self.n_example_pred > 0:
            # For now use val_steps_to_log to determine which steps
            # to make these plots for
            plot_log_steps = list(
                filter(
                    lambda s: s <= target_states.shape[1],
                    self.args.val_steps_to_log,
                )
            )
            # Plot forecasts
            traj_plots = self.plot_ensemble_examples(
                batch,
                n_examples=self.n_example_pred,
                prediction=trajectories,
                time_steps=plot_log_steps,
                log=False,
            )

            # Store plots
            log_plot_dict = {}
            log_plot_dict.update(
                {
                    f"prior_{plot_key}": plot
                    for plot_key, plot in traj_plots.items()
                }
            )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

            plt.close("all")
