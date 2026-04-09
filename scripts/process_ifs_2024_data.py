#!/usr/bin/env python3
"""
Process IFS weather forecasts: merge zarrs and convert 6h timesteps to daily averages.

Input zarrs:
  - zarr1: 10-day forecasts (6h steps, no lead_time=0), one init per Wednesday in 2024
  - zarr2: Analysis fields (lead_time=0 only), same Wednesdays as zarr1
  - zarr3: 24h forecasts (6h steps, lead_time 0-24h), one init per Monday and Tuesday in 2024

Output zarr:
  - init_time: each Tuesday in 2024
  - lead_time: timedelta64 days relative to Tuesday initialization
      -1 day  = Monday daily mean
       0 days = Tuesday daily mean
      +1 day  = Wednesday daily mean (from 10-day forecast)
      2..10 days = Thursday through Friday-of-following-week daily means
"""

import numpy as np
import pandas as pd
import xarray as xr

ZARR_LONG_PATH = "ifs_fc240_subset.zarr"       # 10-day Wednesday forecasts, no lead_time=0
ZARR_ANALYSIS_PATH = "ifs_fc240_analysis_subset_dummy.zarr"  # Analysis (lead_time=0 only), Wednesdays
ZARR_SHORT_PATH = "ifs_fc24_subset.zarr"       # 24h Mon/Tue forecasts
OUTPUT_PATH = "ifs2024_daily_subset.zarr"

# Variables accumulated from forecast start; all others are treated as instantaneous
ACCUM_VARS = ["ssrd", "strd"]
SECONDS_PER_DAY = 86400


def compute_daily_mean_long(ds):
    """Compute daily means for a 10-day, 6-hourly forecast (lead_time 0h to 240h).

    Returns a dataset with timedelta64 lead_time coordinate (1 day to 10 days).

    Instantaneous variables: mean of T+00h, T+06h, T+12h, T+18h for each calendar day.
    Accumulated variables (ssrd, strd): 24h increment / seconds_per_day.
    """
    inst_vars = [v for v in ds.data_vars if v not in ACCUM_VARS]

    # Instantaneous: 4 steps/day × 10 days = 40 steps covering 0h..234h
    hours_inst = np.arange(0, 240, 6)  # [0, 6, 12, ..., 234]
    ds_inst = ds[inst_vars].sel(lead_time=pd.to_timedelta(hours_inst, unit="h"))
    lead_day_labels = pd.to_timedelta(hours_inst // 24 + 1, unit="D")
    ds_inst_daily = (
        ds_inst
        .assign_coords(lead_day=("lead_time", lead_day_labels))
        .groupby("lead_day")
        .mean("lead_time")
        .rename({"lead_day": "lead_time"})
    )

    # Accumulated: 11 daily boundary points (0h, 24h, ..., 240h) → 10 day-increments
    hours_bounds = np.arange(0, 241, 24)
    ds_accum = ds[ACCUM_VARS].sel(lead_time=pd.to_timedelta(hours_bounds, unit="h"))
    ds_accum_daily = (
        (ds_accum.diff(dim="lead_time") / SECONDS_PER_DAY)
        # After diff, lead_time coordinate is [24h, 48h, ..., 240h]; relabel as 1D..10D
        .assign_coords(lead_day=("lead_time", pd.to_timedelta(np.arange(1, 11), unit="D")))
        .swap_dims({"lead_time": "lead_day"})
        .drop_vars("lead_time")
        .rename({"lead_day": "lead_time"})
    )

    return xr.merge([ds_inst_daily, ds_accum_daily])


def compute_daily_mean_short(ds):
    """Compute daily mean for a 24h, 6-hourly forecast (lead_time 0h to 24h).

    Returns a dataset without a lead_time dimension (one daily mean per init_time).

    Instantaneous variables: mean of T+00h, T+06h, T+12h, T+18h.
    Accumulated variables (ssrd, strd): (T+24h - T+00h) / seconds_per_day.
    """
    inst_vars = [v for v in ds.data_vars if v not in ACCUM_VARS]

    lt_inst = pd.to_timedelta([0, 6, 12, 18], unit="h")
    ds_inst_daily = ds[inst_vars].sel(lead_time=lt_inst).mean("lead_time")

    lt0 = pd.Timedelta(hours=0)
    lt24 = pd.Timedelta(hours=24)
    ds_accum_daily = (
        ds[ACCUM_VARS].sel(lead_time=lt24, drop=True)
        - ds[ACCUM_VARS].sel(lead_time=lt0, drop=True)
    ) / SECONDS_PER_DAY

    return xr.merge([ds_inst_daily, ds_accum_daily])


def main():
    print("Loading zarr stores...")
    ds_long = xr.open_zarr(ZARR_LONG_PATH).isel(ensemble=0, drop=True)
    ds_analysis = xr.open_zarr(ZARR_ANALYSIS_PATH).isel(ensemble=0, drop=True)
    ds_short = xr.open_zarr(ZARR_SHORT_PATH).isel(ensemble=0, drop=True)

    # Step 1: Restore lead_time=0 to the long forecast by merging with analysis
    print("Merging long forecast with analysis (restoring lead_time=0)...")
    ds_merged = xr.concat([ds_analysis, ds_long], dim="lead_time").sortby("lead_time")

    # Step 2: Compute daily means for each forecast type
    print("Computing daily means for 10-day (Wednesday) forecasts...")
    ds_long_daily = compute_daily_mean_long(ds_merged)  # lead_time: 1D..10D

    print("Computing daily means for 24h (Monday/Tuesday) forecasts...")
    ds_short_daily = compute_daily_mean_short(ds_short)  # no lead_time dim

    # Step 3: Assemble final dataset with init_time=Tuesday, lead_time=-1D..10D
    print("Assembling final dataset...")
    wednesdays = pd.DatetimeIndex(ds_long_daily.init_time.values)
    tuesdays = wednesdays - pd.Timedelta(days=1)
    mondays = wednesdays - pd.Timedelta(days=2)

    # Select Mon/Tue forecasts and re-label their init_time as the matching Tuesday
    mon_daily = (
        ds_short_daily.sel(init_time=mondays)
        .assign_coords(init_time=tuesdays)
        .expand_dims(lead_time=[pd.Timedelta(days=-1)])
    )
    tue_daily = (
        ds_short_daily.sel(init_time=tuesdays)
        .assign_coords(init_time=tuesdays)
        .expand_dims(lead_time=[pd.Timedelta(days=0)])
    )
    # Re-label long forecast init_time from Wednesday to Tuesday
    ds_long_daily = ds_long_daily.assign_coords(init_time=tuesdays)

    result = xr.concat([mon_daily, tue_daily, ds_long_daily], dim="lead_time")
    result["lead_time"].attrs.update({
        "long_name": "Lead time relative to Tuesday initialization",
    })
    for var in ACCUM_VARS:
        result[var].attrs["units"] = "W m**-2"

    print(f"Saving to {OUTPUT_PATH}...")
    result.to_zarr(OUTPUT_PATH, zarr_format=2)
    print("Done.")


if __name__ == "__main__":
    main()
