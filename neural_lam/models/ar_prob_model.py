# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import torch
import wandb
import xarray as xr
from loguru import logger

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..datastore.base import BaseRegularGridDatastore
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
        datastore_atmosphere: Union[BaseDatastore, None],
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            datastore_atmosphere=datastore_atmosphere,
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

    def sample_trajectories(
        self,
        init_states: torch.Tensor,
        forcing: torch.Tensor,
        boundary_forcing: torch.Tensor,
        atmosphere_forcing: torch.Tensor,
        ensemble_size: int = None,
    ):
        """
        Sample trajectories from the model.
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, num_interior_nodes, d_static_f)
        boundary_forcing: (B, num_boundary_nodes, d_boundary_f)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)

        Returns:
        sampled_trajectories: (num_traj, B, pred_steps, num_interior_nodes, d_f)
        """
        if ensemble_size is None:
            ensemble_size = self.ensemble_size

        traj_list = [
            self.unroll_prediction(
                init_states,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
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
        init_states, target_states, forcing_features, boundary_forcing, atmosphere_forcing, _ = batch

        trajectories = self.sample_trajectories(
            init_states,
            forcing_features,
            boundary_forcing,
            atmosphere_forcing,
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
            atmosphere_forcing,
            time,
        ) = batch

        if prediction is None:
            trajectories = self.sample_trajectories(
                init_states,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
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
