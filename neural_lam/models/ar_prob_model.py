# Standard library
import os
import pickle
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import xarray as xr
from loguru import logger

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..datastore.base import BaseRegularGridDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset
from .ar_model import ARModel


class ARProbModel(ARModel):
    """
    Generic auto-regressive probabilistic weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )

        self.ensemble_size = args.ensemble_size

        self.val_metrics.update(
            {
                "ens_mse": [],
                "spread_squared": [],
            }
        )

        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

    # Training
    def predict_step_train(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        target_state,
    ):
        """
        This method is used during training to predict the next state
        and compute the loss.
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1

        Inputs:
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)

        Returns:
        pred_state: (B, num_interior_nodes, feature_dim)
        pred_std: (B, num_interior_nodes, feature_dim) or None
        loss: (B)
        """
        raise NotImplementedError("No prediction step implemented for training")

    def unroll_prediction_train(
        self, init_states, forcing, boundary_forcing, target_states
    ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, pred_steps, num_interior_nodes, d_static_f)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary_f)

        Returns:
        prediction: (B, pred_steps, num_interior_nodes, d_f)
        pred_std: (B, pred_steps, num_interior_nodes, d_f) or (d_f,)
        loss: (B, pred_steps)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        loss_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            forcing_step = forcing[:, i]

            if self.boundary_forced:
                boundary_forcing_step = boundary_forcing[:, i]
            else:
                boundary_forcing_step = None

            target_state_step = target_states[:, i]

            pred_state, pred_std, loss = self.unroll_ckpt_func(
                self.predict_step_train,
                prev_state,
                prev_prev_state,
                forcing_step,
                boundary_forcing_step,
                target_state_step,
            )
            # state: (B, num_interior_nodes, d_f)
            # pred_std: (B, num_interior_nodes, d_f) or None

            prediction_list.append(pred_state)

            if self.output_std:
                pred_std_list.append(pred_std)

            loss_list.append(loss)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_interior_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_interior_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        loss = torch.stack(
            loss_list, dim=1
        )  # (B, pred_steps, num_interior_nodes, d_f)

        return prediction, pred_std, loss

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
            _,
        ) = batch

        _, _, loss = self.unroll_prediction_train(
            init_states, forcing, boundary_forcing, target_states
        )
        # prediction: (B, pred_steps, num_interior_nodes, d_f)
        # pred_std: (B, pred_steps, num_interior_nodes, d_f) or (d_f,)
        # loss: (B, pred_steps)

        # Compute loss - mean over unrolled times and batch
        batch_loss = torch.mean(loss)

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

    # Evaluation
    def predict_step(
        self, prev_state, prev_prev_state, forcing, boundary_forcing
    ):
        """
        This method is used during inference to sample a prediction of the next state.
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1

        Inputs:
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)

        Returns:
        pred_state: (B, num_interior_nodes, feature_dim)
        pred_std: (B, num_interior_nodes, feature_dim) or None
        """
        raise NotImplementedError(
            "No prediction step implemented for inference"
        )

    def sample_trajectories(
        self,
        init_states: torch.Tensor,
        forcing: torch.Tensor,
        boundary_forcing: torch.Tensor,
    ):
        """
        Sample trajectories from the model.
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, num_interior_nodes, d_static_f)
        boundary_forcing: (B, num_boundary_nodes, d_boundary_f)

        Returns:
        sampled_trajectories: (num_traj, B, pred_steps, num_interior_nodes, d_f)
        """

        traj_list = [
            self.unroll_prediction(
                init_states,
                forcing,
                boundary_forcing,
            )
            for _ in range(self.ensemble_size)
        ]

        traj_tensor = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )

        return traj_tensor

    def ensemble_step(self, batch):
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
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            _,
        ) = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            None,
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
            sum_vars=False,
        )  # (B, pred_steps, d_f)

        return (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        self.log_spsk_ratio(self.val_metrics, "val")
        super().on_validation_epoch_end()

    def _save_predictions_to_zarr(
        self,
        batch_times: torch.Tensor,
        batch_predictions: torch.Tensor,
        batch_idx: int,
        zarr_output_path: str,
    ):
        """
        Save state predictions for single batch to zarr dataset. Will append to
        existing dataset for batch_idx > 0. Resulting dataset will contain a
        variable named `state` with coordinates (start_time,
        elapsed_forecast_duration, grid_index, state_feature).
        Parameters
        ----------
        batch_times : torch.Tensor[int]
            The times for the batch, given as epoch time in nanoseconds. Shape
            is (B, args.pred_steps) where B is the batch size and
            args.pred_steps is the number of prediction steps.
        batch_predictions : torch.Tensor[float]
            The predictions for the batch, given as (B, args.pred_steps,
            num_grid_nodes, d_f) where B is the batch size, args.pred_steps is
            the number of prediction steps, num_grid_nodes is the number of
            grid nodes, and d_f is the number of state features.
        batch_idx : int
            The index of the batch in the current epoch.
        """
        # Scale predictions back to original data scale
        batch_predictions_rescaled = (
            batch_predictions * self.state_std + self.state_mean
        )

        # Convert predictions to DataArray using _create_dataarray_from_tensor
        das_pred = []
        for i in range(len(batch_times)):
            da_pred = self._create_dataarray_from_tensor(
                tensor=batch_predictions_rescaled[i],
                time=batch_times[i],
                split="test",
                category="state",
            )
            # Unstack grid coords if necessary, this also avoids the need to
            # try to store a MultiIndex zarr dataset which is not supported by
            # xarray
            if isinstance(self._datastore, BaseRegularGridDatastore):
                da_pred = self._datastore.unstack_grid_coords(da_pred)

            # First entry in da_pred.coords["time"] is time of first prediction,
            # so init time of forecast is one time step before
            t0 = da_pred.coords["time"].values[0] - np.array(
                self.step_length, dtype="timedelta64[h]"
            )
            da_pred.coords["start_time"] = t0
            da_pred.coords["elapsed_forecast_duration"] = da_pred.time - t0
            da_pred = da_pred.swap_dims({"time": "elapsed_forecast_duration"})
            da_pred.name = "state"
            das_pred.append(da_pred)

        da_pred_batch = xr.concat(das_pred, dim="start_time")

        # Apply chunking start_time and elapsed_forecast_duration, but leave
        # whole state in one chunk
        da_pred_batch = da_pred_batch.chunk(
            {"start_time": 1, "elapsed_forecast_duration": 1}
        )

        if batch_idx == 0:
            logger.info(f"Saving predictions to {zarr_output_path}")
            compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
            )
            da_pred_batch.to_zarr(
                zarr_output_path,
                mode="w",
                consolidated=True,
                encoding={
                    "start_time": {
                        "units": "Seconds since 1970-01-01 00:00:00",
                        "dtype": "int64",
                    },
                    "state": {"compressor": compressor},
                },
            )
        else:
            da_pred_batch.to_zarr(
                zarr_output_path, mode="a", append_dim="start_time"
            )

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        super().test_step(batch, batch_idx)

        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_step(batch)

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
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

        if self.args.save_eval_to_zarr_path:
            self._save_predictions_to_zarr(
                batch_times=batch[-1],
                batch_predictions=trajectories,
                batch_idx=batch_idx,
                zarr_output_path=self.args.save_eval_to_zarr_path,
            )

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                trajectories.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_ensemble_examples(
                batch,
                n_additional_examples,
                prediction=trajectories,
                split="test",
            )
            self.plotted_examples += n_additional_examples

    def plot_examples(self, batch, n_examples, split, prediction=None):
        # We don't want to plot examples for a single ensemble member.
        pass

    def plot_ensemble_examples(
        self,
        batch,
        n_examples,
        split="train",
        prediction=None,
        time_steps=None,
        log=True,
    ):
        """
        Plot ensemble forecast + mean and std
        (split argument should be unused, only for compatibility with ARModel)
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            time,
        ) = batch

        if prediction is None:
            trajectories = self.sample_trajectories(
                init_states,
                forcing,
                boundary_forcing,
            )
            # (B, S, pred_steps, num_grid_nodes, d_f)
        else:
            trajectories = prediction
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
        for example_i, (
            traj_slice,
            target_slice,
            ens_mean_slice,
            ens_std_slice,
            time_slice,
        ) in enumerate(
            zip(
                traj_rescaled[:n_examples],
                target_rescaled[:n_examples],
                ens_mean[:n_examples],
                ens_std[:n_examples],
                time[:n_examples],
            ),
            start=1,
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

            if time_steps is None:
                # If no set of time steps given, iterate over all
                # prediction horizon time steps
                time_steps = range(1, len(target_slice) + 1)

            plot_dict = {}
            for t_i in time_steps:
                time_title_part = (
                    f"t={t_i} ({self._datastore.step_length*t_i} h)"
                )
                # Create one figure per variable at this time step
                var_figs = {
                    var_name: vis.plot_ensemble_prediction(
                        [
                            da_samples[i].isel(
                                state_feature=var_i, time=t_i - 1
                            )
                            for i in range(traj_slice.shape[0])
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
                    # Only plot self.plot_vars
                    if var_name in self.plot_vars
                }

                if log:
                    # Log immediately
                    example_title = f"example_{example_i}"
                    wandb.log(
                        {
                            f"{var_name}_{example_title}": wandb.Image(fig)
                            for var_name, fig in var_figs.items()
                        }
                    )
                else:
                    # Store to return
                    plot_dict.update(
                        {
                            f"{var_name}_step_{t_i}_ex{example_i}": wandb.Image(
                                fig
                            )
                            for var_name, fig in var_figs.items()
                        }
                    )

                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

        return plot_dict

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        super().on_test_epoch_end()
        self.log_spsk_ratio(self.test_metrics, "test")

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
