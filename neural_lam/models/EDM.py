# Standard library
import math

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# First-party
from neural_lam import metrics, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class EDM(ARModel):
    """
    Diffusion Forecasting Model using the EDM framework.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
    ):
        super().__init__(args, config, datastore)

        # ----------------------------------------------------------------------------
        # Diffusion (EDM) parameters
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.sigma_data = args.sigma_data
        self.rho = args.rho
        self.sampler = args.sampler
        self.sampler_steps = args.sampler_steps
        self.ensemble_size = args.ensemble_size
        self.pred_residual = (
            args.pred_residual
        )  # Whether to predict the residual instead of the next state

        if args.diffusion_model == "graph_diff":
            print("Using GraphDiff")
            self.model = GraphDiff(args, config, datastore)
        else:
            raise NotImplementedError(
                f"Unknown diffusion model: {args.diffusion_model}"
            )

        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

    # ----------------------------------------------------------------------------
    # EDM model methods
    def denoise(self, x, sigma, class_labels=None, **model_kwargs):
        """""
        Denoising forward pass through the diffusion backbone model.
        x: (B, N_grid, d_state)
        sigma: (B), noise level
        class_labels: (B, N_grid, d_conditioning)
        model_kwargs: additional arguments for the model

        Returns:
        D_x: (B, N_grid, d_state), denoised state
        """ ""
        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x), c_noise.flatten(), class_labels, **model_kwargs
        )
        D_x = c_skip * x + c_out * F_x

        return D_x

    # Evaluation
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
        input_grid = torch.cat(
            (prev_state, prev_prev_state, forcing), dim=-1
        )  # (B, N_grid, d_input)

        latents = torch.randn_like(prev_state)  # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            next_state = self.heun_sampler(
                latents=latents,
                class_labels=input_grid,
                sigma_min=self.sigma_min * 1.5,
            )
        elif self.sampler == "edm":
            next_state = self.edm_sampler(
                latents=latents,
                class_labels=input_grid,
                sigma_min=self.sigma_min * 1.5,
                num_steps=self.sampler_steps,
            )
        elif self.sampler == "ddpm":
            next_state = self.ddpm_sampler(
                latents=latents,
                class_labels=input_grid,
                sigma_min=self.sigma_min * 1.5,
            )

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        return next_state

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)

        Returns:
        prediction: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            true_state = true_states[:, i]
            pred_state = self.predict_step(prev_state, prev_prev_state, forcing)

            # Overwrite border with true state
            pred_state = (
                self.boundary_mask * true_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(pred_state)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        return prediction, self.per_var_std

    def sample_trajectories(
        self,
        init_states,
        forcing_features,
        target_states,
        num_traj,
    ):
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        num_traj: S, number of trajectories to sample

        Returns
        traj_tensor: (B, S, pred_steps, num_grid_nodes, d_f)
        """

        traj_list = [
            self.unroll_prediction(
                init_states,
                forcing_features,
                target_states,
            )
            for _ in range(num_traj)
        ]

        traj_tensor = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )

        return traj_tensor

    def ensemble_common_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Common step done during both evaluation and testing

        batch: tuple of tensors, batch to perform ensemble forecast on

        Returns:
        trajectories: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f)
        target_states: (B, pred_steps, num_grid_nodes, d_f)
        spread_squared_batch: (B, pred_steps, d_f)
        ens_mse_batch: (B, pred_steps, d_f)
        """
        # Compute and store metrics for ensemble forecast
        init_states, target_states, forcing_features, _ = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            None,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        # (B, pred_steps, d_f)

        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)

        return (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot ensemble forecast + mean and std
        (split argument should be unused, only for compatibility with ARModel)
        """
        init_states, target_states, forcing_features, time = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.state_std + self.state_mean
        target_rescaled = target_states * self.state_std + self.state_mean

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for (
            traj_slice,
            target_slice,
            ens_mean_slice,
            ens_std_slice,
            time_slice,
        ) in zip(
            traj_rescaled[:n_examples],
            target_rescaled[:n_examples],
            ens_mean[:n_examples],
            ens_std[:n_examples],
            time[:n_examples],
        ):

            # Create xarray for plotting
            da_samples = [
                self._create_dataarray_from_tensor(
                    tensor=traj_slice[i, ...],
                    time=time_slice,
                    split=split,
                    category="state",
                ).unstack("grid_index")
                for i in range(traj_slice.shape[0])
            ]

            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_mean = self._create_dataarray_from_tensor(
                tensor=ens_mean_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            da_ens_std = self._create_dataarray_from_tensor(
                tensor=ens_std_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (samples_t, target_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
                    target_slice,
                    ens_mean_slice,
                    ens_std_slice,
                ),
                start=1,
            ):
                time_title_part = (
                    f"t={t_i} ({self._datastore.step_length*t_i} h)"
                )
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_ensemble_prediction(
                        [
                            da_samples[i].isel(
                                state_feature=var_i, time=t_i - 1
                            )
                            for i in range(traj_slice.shape[1])
                        ],
                        da_target.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_mean.isel(state_feature=var_i, time=t_i - 1),
                        da_ens_std.isel(state_feature=var_i, time=t_i - 1),
                        self._datastore,
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                if (
                    self.trainer.is_global_zero
                    and not self.trainer.sanity_checking
                ):
                    current_epoch = self.trainer.current_epoch
                else:
                    current_epoch = "NAN"

                example_title = (
                    f"example_{self.plotted_examples}_epoch_{current_epoch}"
                )

                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(
                            self._datastore.get_vars_names("state"), var_figs
                        )
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def log_spsk_ratio(self, metric_vals, prefix):
        """
        Compute the mean spread-skill ratio for logging in evaluation

        metric_vals: dict with all metric values
        prefix: string, prefix to use for logging
        """
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )  # (N_eval, pred_steps, d_f)
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )  # (N_eval, pred_steps, d_f)

        # Do not log during sanity check?
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))
            # Both (pred_steps, d_f)

            # Include finite sample correction
            spsk_ratios = np.sqrt(
                (self.ensemble_size + 1) / self.ensemble_size
            ) * (
                spread / skill
            )  # (pred_steps, d_f)
            log_dict = self.create_metric_log_dict(
                spsk_ratios, prefix, "spsk_ratio"
            )

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(
                spsk_ratios
            )  # log mean
            wandb.log(log_dict)

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """
        (
            init_states,
            target_states,
            forcing_features,
            _,
        ) = batch

        super().test_step(batch, batch_idx)

        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)
        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # super().on_test_epoch_end()
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")
        self.log_spsk_ratio(self.test_metrics, "test")

    # Validation
    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        # TODO: Validation step (calculate loss), if it is meaningful?
        # Validation step batch 0, sample 1 trajectory for visual evaluation
        # Plot some example predictions using prior and encoder
        val_log_dict = {
            "val_mean_loss": torch.tensor([0], device=batch[0].device)
        }
        batch_idx = args[0]
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.n_example_pred > 0
        ):
            (
                trajectories,
                target_states,
                spread_squared_batch,
                ens_mse_batch,
            ) = self.ensemble_common_step(batch)
            # We only take the statistics from the plotted samples as we will
            # not sample for the whole validation set
            # NOTE: This metric is not that useful,
            # as we only sample 1 trajectory
            val_log_dict["val_mean_loss"] = ens_mse_batch.mean()
            self.plot_examples(
                batch,
                n_examples=self.n_example_pred,
                split="val",
                prediction=trajectories,
            )
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Must log before super call, as metric lists are cleared at end of step
        # super().on_validation_epoch_end()
        print("End of validation epoch")
        # We don't save any validation metrics for now so we want to skip this

    # Training
    def predict_step_train(
        self, prev_state, prev_prev_state, forcing, target_state
    ):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static
        forcing: (B, N_grid, forcing_dim), dynamic forcing

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at t+1
        loss: (B)
        """

        # Sample from F inverse
        rnd_uniform = torch.rand(
            [prev_state.shape[0], 1, 1], device=prev_state.device
        )
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max**rho_inv
        sigma_min_rho = self.sigma_min**rho_inv
        sigma = (
            sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho

        input_grid = torch.cat((prev_state, prev_prev_state, forcing), dim=-1)

        # Make y residual if needed
        if self.pred_residual:
            y = target_state - prev_state
            y = (
                target_state - self.diff_mean
            ) / self.diff_std  # Normalize residual

        n = torch.randn_like(y) * sigma
        noisy_input = y + n

        next_state = self.denoise(
            noisy_input, sigma, input_grid
        )  # Shape (B, d_state, N_x, N_y)

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        # Calculate the loss
        # Weights for the loss function based on the noise level
        weight = (
            (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        ).squeeze()  # (B)

        pred_std = self.per_var_std

        loss = self.loss(
            next_state,
            target_state,
            pred_std,
            mask=self.interior_mask_bool,
        )  # (B)

        loss = loss * weight.squeeze()  # (B)

        return next_state, loss

    def unroll_prediction_train(
        self, init_states, forcing_features, true_states
    ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)

        Returns:
        prediction: (B, pred_steps, num_grid_nodes, d_f)
        loss_tensor: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_steps = forcing_features.shape[1]
        loss_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            true_state = true_states[:, i]

            pred_state, loss = self.predict_step_train(
                prev_state, prev_prev_state, forcing, true_state
            )

            # Overwrite border with true state
            pred_state = (
                self.boundary_mask * true_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(pred_state)
            loss_list.append(loss)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        loss_tensor = torch.stack(loss_list, dim=1)  # (B)

        return prediction, loss_tensor

    def common_step_train(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """

        (init_states, target_states, forcing, _) = batch
        prediction, loss = self.unroll_prediction_train(
            init_states, forcing, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, loss = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(loss)  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction,
                target,
                mask=self.interior_mask_bool,
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss, "train_mse": batch_mse}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # This work is licensed under a Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # You should have received a copy of the license along with this
    # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed EDM sampler (Algorithm 2).

    def edm_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        randn_like=torch.randn_like,
        num_steps=20,
        sigma_min=0.03,
        sigma_max=80,
        rho=7,
        S_churn=2.5,
        S_min=0.75,
        S_max=80,
        S_noise=1.05,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [
                torch.as_tensor(t_steps, device=latents.device),
                torch.zeros_like(t_steps[:1], device=latents.device),
            ]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next
            # diff_steps.append(x_cur)

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(
                t_cur + gamma * t_cur, device=latents.device
            )
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * randn_like(x_cur, device=latents.device)

            # Euler step.
            denoised = self.denoise(x_hat, t_hat, class_labels=class_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next, t_next, class_labels=class_labels
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        randn_like=torch.randn_like,
        num_steps=20,
        sigma_min=0.03,
        sigma_max=80,
        rho=7,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [
                torch.as_tensor(t_steps, device=latents.device),
                torch.zeros_like(t_steps[:1], device=latents.device),
            ]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next
            denoised = self.denoise(x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next, t_next, class_labels=class_labels
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    # ----------------------------------------------------------------------------

    # Sampler used in GenCast (taken from a reimplementation of the paper
    # before the official code was released).
    # TODO: Check if this is correct.
    def ddpm_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        randn_like=torch.randn_like,
        num_steps=20,
        sigma_min=0.03,
        sigma_max=80,
        rho=7,
        S_churn=2.5,
        S_min=0.75,
        S_max=80,
        S_noise=1.05,
        r=0.5,
    ):

        time_steps = torch.arange(0, num_steps, device=latents.device) / (
            num_steps - 1
        )
        sigmas = (
            sigma_max ** (1 / rho)
            + time_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

        # initialize noise
        x = sigmas[0] * latents

        for i in range(len(sigmas) - 1):
            # stochastic churn from Karras et al. (Alg. 2)
            gamma = (
                min(S_churn / num_steps, math.sqrt(2) - 1)
                if S_min <= sigmas[i] <= S_max
                else 0.0
            )
            # noise inflation from Karras et al. (Alg. 2)
            noise = S_noise * randn_like(latents, device=latents.device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = self.denoise(x, sigma_hat, class_labels=class_labels)

            if i == len(sigmas) - 2:
                # final Euler step
                d = (x - denoised) / sigma_hat
                x = x + d * (sigmas[i + 1] - sigma_hat)
            else:
                # DPMSolver++2S  step (Alg. 1 in Lu et al.) with alpha_t=1.
                # t_{i-1} is t_hat because of stochastic churn!
                lambda_hat = -torch.log(sigma_hat)
                lambda_next = -torch.log(sigmas[i + 1])
                h = lambda_next - lambda_hat
                lambda_mid = lambda_hat + r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = (
                    sigma_mid / sigma_hat * x
                    - (torch.exp(-r * h) - 1) * denoised
                )
                denoised_2 = self.denoise(
                    u, sigma_mid, class_labels=class_labels
                )
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x
