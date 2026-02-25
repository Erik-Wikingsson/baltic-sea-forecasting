# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import torch
import wandb

# First-party
from neural_lam import metrics

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from .ar_prob_model import ARProbModel
from .graph_diff import GraphDiff


class Drifting(ARProbModel):
    """
    Drifting model.

    Training uses a fixed-point drifting objective in data space:
        L = ||x - stopgrad(x + V)||^2
    with:
        - attraction drift toward the target state (positive samples),
        - repulsion drift away from batch ensemble mean (negative samples).
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
            args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
        )

        if args.backbone_model == "graph_diff":
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

        self.val_metrics.update({"ens_crps": []})

    def _masked_mse(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE reduced with the same interior-mask convention as other models.
        Supports tensors of shape (B, N, d_state) or (B, S, N, d_state).
        """
        sq_err = (prediction - target) ** 2
        if sq_err.ndim == 4:
            bsz, n_ens, n_grid, d_state = sq_err.shape
            sq_err = sq_err.reshape(bsz * n_ens, n_grid, d_state)
        per_sample = metrics.mask_and_reduce_metric(
            sq_err,
            mask=self.interior_mask_bool,
            average_grid=True,
            sum_vars=True,
        )
        return torch.mean(per_sample)

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        One-step stochastic forecast used during autoregressive rollout/eval.
        """
        input_grid = torch.cat(
            (prev_state, prev_prev_state, forcing), dim=-1
        )  # (B, N_grid, d_input)

        # Match CRPS: 2D noise tensor of shape (B, noise_dim).
        z = torch.randn(
            prev_state.shape[0], self.model.noise_dim, device=prev_state.device
        )

        next_state = self.model(
            input_grid,
            noise_level=z,
            cond=None,
            boundary_forcing=boundary_forcing,
            atmosphere_forcing=atmosphere_forcing,
        )  # (B, N_grid, d_state)

        if self.args.pred_residual:
            next_state = (next_state * self.diff_std) + self.diff_mean
            next_state = prev_state + next_state

        return next_state, self.per_var_std

    def training_step(self, batch):
        """
        Train with autoregressive unrolling and data-space drifting objectives.
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch

        # Sample trajectories
        trajectories = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            ensemble_size=self.ensemble_size,
        )  # (B, S, pred_steps, N_grid, d_state)

        target_expanded = target_states.unsqueeze(
            1
        )  # (B, 1, pred_steps, N_grid, d_state)

        # Attraction drift (pos. samples): pull members toward truth.
        attraction_target = (
            trajectories + (target_expanded - trajectories)
        ).detach()
        mean_attraction = self._masked_mse(trajectories, attraction_target)

        # Repulsion drift (neg. samples): push members away from ensemble mean.
        member_mean = trajectories.mean(dim=1, keepdim=True)
        repulsion_target = (
            trajectories + (trajectories - member_mean)
        ).detach()
        mean_repulsion = self._masked_mse(trajectories, repulsion_target)

        loss = mean_attraction + mean_repulsion

        self.log_dict(
            {
                "train_loss": loss,
                "drift_attraction_loss": mean_attraction,
                "drift_repulsion_loss": mean_repulsion,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def validation_step(self, batch, *args):
        """
        Run validation on single batch and log ensemble plots like CRPS.
        """
        super().validation_step(batch, *args)
        batch_idx = args[0]
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

        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.n_example_pred > 0
        ):
            plot_log_steps = list(
                filter(
                    lambda s: s <= target_states.shape[1],
                    self.args.val_steps_to_log,
                )
            )
            traj_plots = self.plot_ensemble_examples(
                batch,
                n_examples=self.n_example_pred,
                prediction=trajectories,
                time_steps=plot_log_steps,
                log=False,
            )
            log_plot_dict = {
                f"prior_{plot_key}": plot
                for plot_key, plot in traj_plots.items()
            }

            if not self.trainer.sanity_checking:
                wandb.log(log_plot_dict)

            plt.close("all")
